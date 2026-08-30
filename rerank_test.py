import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
import jieba

API_KEY = "9f8b164567c54051aa88b29dd4cf11f3.2Bz5nd5sxvCxCOiU"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DOCS_DIR = os.path.expanduser("~/ecommerce_rag/data/docs")
CHROMA_DIR = os.path.expanduser("~/ecommerce_rag/chroma_db")
RERANKER_PATH = "/home/lzj/.cache/modelscope/models/BAAI--bge-reranker-base/snapshots/master"

# 加载+分块
print("【1】加载文档...")
loader = DirectoryLoader(DOCS_DIR, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 向量检索
embeddings = OpenAIEmbeddings(model="embedding-2", openai_api_key=API_KEY, openai_api_base=BASE_URL)
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# BM25（jieba中文分词）
def chinese_preprocess(text):
    return list(jieba.cut(text))
bm25_retriever = BM25Retriever.from_documents(chunks, preprocess_func=chinese_preprocess)
bm25_retriever.k = 10

# 混合检索（RRF融合）
def hybrid_search(query, k=10):
    bm25_docs = bm25_retriever.invoke(query)
    vec_docs = vector_retriever.invoke(query)
    rrf = {}
    all_docs = {}
    for rank, doc in enumerate(bm25_docs):
        key = hash(doc.page_content)
        rrf[key] = rrf.get(key, 0) + 1/(60 + rank + 1)
        all_docs[key] = doc
    for rank, doc in enumerate(vec_docs):
        key = hash(doc.page_content)
        rrf[key] = rrf.get(key, 0) + 1/(60 + rank + 1)
        if key not in all_docs:
            all_docs[key] = doc
    sorted_keys = sorted(rrf.keys(), key=lambda x: rrf[x], reverse=True)
    return [all_docs[k] for k in sorted_keys[:k]]

# 加载bge-reranker
print("【2】加载bge-reranker模型（首次加载稍慢）...")
reranker = CrossEncoder(RERANKER_PATH)
print("    加载完成")

def rerank_docs(query, docs, top_n=3):
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked[:top_n]]

# 测试对比
print("\n【3】测试对比：\n")
questions = [
    "iPhone 15的电池容量是多少？",
    "退货政策是什么？几天可以退？",
    "iPhone 15和华为Mate60哪个电池大？",
]

for q in questions:
    print("=" * 60)
    print(f"问题：{q}")
    print("\n  混合检索top5：")
    ens_docs = hybrid_search(q, k=10)[:5]
    for i, d in enumerate(ens_docs, 1):
        print(f"    {i}. {os.path.basename(d.metadata.get('source',''))}")
    print("\n  混合+Reranker top3：")
    all_ens = hybrid_search(q, k=10)
    reranked = rerank_docs(q, all_ens, top_n=3)
    for i, d in enumerate(reranked, 1):
        print(f"    {i}. {os.path.basename(d.metadata.get('source',''))}")
    print()
