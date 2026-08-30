import os, json
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
import jieba

API_KEY = "9f8b164567c54051aa88b29dd4cf11f3.2Bz5nd5sxvCxCOiU"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DOCS_DIR = os.path.expanduser("~/ecommerce_rag/data/docs")
CHROMA_DIR = os.path.expanduser("~/ecommerce_rag/chroma_db")
RERANKER_PATH = "/home/lzj/.cache/modelscope/models/BAAI--bge-reranker-base/snapshots/master"

llm = ChatOpenAI(model="glm-4-flash", openai_api_key=API_KEY, openai_api_base=BASE_URL, temperature=0)
embeddings = OpenAIEmbeddings(model="embedding-2", openai_api_key=API_KEY, openai_api_base=BASE_URL)

print("加载检索组件...")
loader = DirectoryLoader(DOCS_DIR, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(docs)
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
def chinese_preprocess(text):
    return list(jieba.cut(text))
bm25_retriever = BM25Retriever.from_documents(chunks, preprocess_func=chinese_preprocess)
bm25_retriever.k = 10
reranker = CrossEncoder(RERANKER_PATH)

def hybrid_search(query, k=3):
    bm25_docs = bm25_retriever.invoke(query)
    vec_docs = vector_retriever.invoke(query)
    rrf, all_docs = {}, {}
    for rank, doc in enumerate(bm25_docs):
        key = hash(doc.page_content)
        rrf[key] = rrf.get(key, 0) + 1/(60+rank+1)
        all_docs[key] = doc
    for rank, doc in enumerate(vec_docs):
        key = hash(doc.page_content)
        rrf[key] = rrf.get(key, 0) + 1/(60+rank+1)
        if key not in all_docs:
            all_docs[key] = doc
    sorted_keys = sorted(rrf.keys(), key=lambda x: rrf[x], reverse=True)
    candidates = [all_docs[k] for k in sorted_keys[:10]]
    pairs = [(query, d.page_content) for d in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked[:k]]

# 自建评估函数（基于RAGAS 4指标，用LLM打分）
def llm_score(prompt):
    result = llm.invoke(prompt).content.strip()
    for val in ["1.0", "0.5", "0.0"]:
        if val in result:
            return float(val)
    return 0.5

def eval_faithfulness(answer, contexts):
    return llm_score(f"判断答案是否忠于文档（没编造信息）。\n文档：{contexts[:800]}\n答案：{answer}\n\n忠于文档返回1.0，有编造返回0.0，部分忠于返回0.5。只返回数字。")

def eval_answer_relevancy(question, answer):
    return llm_score(f"判断答案是否回答了问题。\n问题：{question}\n答案：{answer}\n\n完全回答返回1.0，没回答返回0.0，部分回答返回0.5。只返回数字。")

def eval_context_precision(question, contexts):
    return llm_score(f"判断检索到的文档是否与问题相关。\n问题：{question}\n文档：{contexts[:800]}\n\n相关返回1.0，不相关返回0.0，部分相关返回0.5。只返回数字。")

def eval_context_recall(ground_truth, contexts):
    return llm_score(f"判断标准答案的信息是否能在文档中找到。\n标准答案：{ground_truth}\n文档：{contexts[:800]}\n\n能找到返回1.0，找不到返回0.0，部分找到返回0.5。只返回数字。")

# 加载测试集
with open(os.path.expanduser("~/ecommerce_rag/test_set.json")) as f:
    test_set = json.load(f)
print(f"测试集：{len(test_set)}题")

# 对每题：检索+生成+评估
scores = {"faithfulness": [], "answer_relevancy": [], "context_precision": [], "context_recall": []}
eval_data = []

for i, item in enumerate(test_set, 1):
    q = item["question"]
    gt = item["ground_truth"]
    print(f"\n[{i}/{len(test_set)}] {q}")
    docs = hybrid_search(q, k=3)
    contexts_str = "\n\n".join(d.page_content for d in docs)
    contexts_list = [d.page_content for d in docs]
    answer = llm.invoke(f"基于以下文档回答问题。文档没有就说'无法回答'。\n\n文档：\n{contexts_str}\n\n问题：{q}").content
    print(f"  答案：{answer[:60]}")
    # 评估4指标
    f_score = eval_faithfulness(answer, contexts_str)
    ar_score = eval_answer_relevancy(q, answer)
    cp_score = eval_context_precision(q, contexts_str)
    cr_score = eval_context_recall(gt, contexts_str)
    scores["faithfulness"].append(f_score)
    scores["answer_relevancy"].append(ar_score)
    scores["context_precision"].append(cp_score)
    scores["context_recall"].append(cr_score)
    print(f"  指标：faith={f_score} rel={ar_score} prec={cp_score} recall={cr_score}")
    eval_data.append({"question": q, "answer": answer, "ground_truth": gt, "scores": {"faithfulness": f_score, "answer_relevancy": ar_score, "context_precision": cp_score, "context_recall": cr_score}})

# 输出结果
print("\n" + "="*50)
print("=== RAGAS评估结果（自建，20题平均）===")
print("="*50)
for k, v in scores.items():
    avg = sum(v) / len(v)
    print(f"  {k}: {avg:.2f}")
print("="*50)

# 保存
with open(os.path.expanduser("~/ecommerce_rag/eval_results.json"), "w") as f:
    json.dump({"scores": {k: sum(v)/len(v) for k,v in scores.items()}, "eval_data": eval_data}, f, ensure_ascii=False, indent=2)
print("结果已保存到 eval_results.json")
