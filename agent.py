import os, time, logging, json, threading
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator, jieba

# ---------- 路径：全部相对本文件，本地/容器通用 ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

API_KEY = os.environ.get("ZHIPU_API_KEY", "")
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DOCS_DIR = os.path.join(BASE_DIR, "data", "docs")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
HISTORY_FILE = os.path.join(BASE_DIR, "qa_history.json")
DOC_VERSION_FILE = os.path.join(BASE_DIR, "doc_version.txt")

# 记忆有效期：7天；知识库版本变化后旧记忆自动失效
MEMORY_TTL = 7 * 24 * 3600

logging.basicConfig(level=logging.INFO, format='%(message)s')
llm = ChatOpenAI(model="glm-4-flash", openai_api_key=API_KEY, openai_api_base=BASE_URL, temperature=0)

# ---------- 记忆：加载/保存（带锁 + 过期 + 版本） ----------
_hist_lock = threading.Lock()

def _doc_version():
    try:
        v = open(DOC_VERSION_FILE).read().strip()
        return v if v else "1"
    except Exception:
        return "1"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(question, answer):
    with _hist_lock:
        history = load_history()
        now = time.time()
        # 顺手清理过期记忆，防止文件越滚越大
        history = [h for h in history if now - h.get("ts", 0) < MEMORY_TTL]
        history.append({"question": question, "answer": answer,
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "ts": now, "doc_version": _doc_version()})
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    logging.info(f"[记忆] 已保存（共{len(history)}条）")

def find_memory(question):
    """命中条件：问题匹配 + 版本一致 + 未过期。旧格式条目没有 ts 视为过期。"""
    now = time.time()
    for item in load_history():
        hq = item.get("question", "")
        matched = (question == hq
                   or (len(question) > 5 and question in hq)
                   or (len(hq) > 5 and hq in question))
        if (matched
                and item.get("doc_version") == _doc_version()
                and now - item.get("ts", 0) < MEMORY_TTL):
            return item
    return None

# ---------- 检索组件 ----------
print("加载检索组件...")
loader = DirectoryLoader(DOCS_DIR, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(docs)
embeddings = OpenAIEmbeddings(model="embedding-2", openai_api_key=API_KEY, openai_api_base=BASE_URL)
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

# 容器首次启动时向量库是空的：自动入库（智谱 embedding 单次最多64条，按50分批）
try:
    _existing = vectorstore.get().get("ids") or []
except Exception:
    _existing = []
if len(_existing) == 0:
    print(f"向量库为空，首次入库（{len(chunks)} 块）...")
    for i in range(0, len(chunks), 50):
        vectorstore.add_documents(chunks[i:i+50])
    print("入库完成。")

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

def chinese_preprocess(text):
    return list(jieba.cut(text))
bm25_retriever = BM25Retriever.from_documents(chunks, preprocess_func=chinese_preprocess)
bm25_retriever.k = 10

# ---------- reranker：本地有就用本地，容器里自动从国内镜像下载 ----------
def _resolve_reranker_path():
    candidates = [
        os.environ.get("RERANKER_PATH", ""),
        "/home/lzj/.cache/modelscope/models/BAAI--bge-reranker-base/snapshots/master",
        os.path.join(BASE_DIR, ".cache_model", "bge-reranker-base"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    print("本地未找到 reranker，从国内镜像下载（约1.2G，仅首次）...")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from huggingface_hub import snapshot_download
    return snapshot_download("BAAI/bge-reranker-base",
                             local_dir=os.path.join(BASE_DIR, ".cache_model", "bge-reranker-base"),
                             allow_patterns=["*.json", "*.txt", "model.safetensors", "tokenizer*"])

reranker = CrossEncoder(_resolve_reranker_path())

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
    candidates = [all_docs[k2] for k2 in sorted_keys[:10]]
    pairs = [(query, d.page_content) for d in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked[:k]]

# ---------- Agent 状态 ----------
class AgentState(TypedDict):
    question: str
    tool: str
    tool_results: Annotated[list, operator.add]
    answer: str
    needs_retry: bool
    retry_count: int
    memory_hit: bool

# 节点1：意图分类（先查记忆，命中直接返回）
def classify_intent(state):
    start = time.time()
    question = state["question"]

    hit = find_memory(question)
    if hit:
        logging.info(f"\n[记忆] 命中历史问答，直接返回（0s）")
        return {"tool": "memory", "answer": hit["answer"], "memory_hit": True}

    prompt = f"""判断这个问题应该用哪个工具：
- rag: 需要查文档知识（商品规格/电池容量/退货政策/运营流程/商品对比/价格/参数）
- calculator: 需要纯数学计算（加减乘除/百分比，不涉及查文档）
- web_search: 需要实时外部信息（天气/新闻/实时股价）

注意：商品信息（电池/价格/规格/参数）在知识库文档里有，用rag，不要用web_search。

问题：{question}
只返回rag、calculator或web_search其中一个词。"""
    result = llm.invoke(prompt).content.strip().lower()
    if "calc" in result:
        tool = "calculator"
    elif "web" in result or "search" in result:
        tool = "web_search"
    else:
        tool = "rag"
    logging.info(f"\n[意图分类] {question} -> {tool}（{time.time()-start:.2f}s）")
    return {"tool": tool, "memory_hit": False}

# 节点2a：RAG检索
def rag_tool(state):
    start = time.time()
    logging.info(f"[RAG检索] 检索中...")
    docs = hybrid_search(state["question"], k=3)
    content = "\n\n".join(d.page_content for d in docs)
    sources = [os.path.basename(d.metadata.get('source','')) for d in docs]
    logging.info(f"[RAG检索] 来源：{sources}（{time.time()-start:.2f}s）")
    return {"tool_results": [{"content": content, "sources": sources}]}

# 节点2b：计算器
def calculator_tool(state):
    start = time.time()
    logging.info(f"[计算器] 计算中...")
    result = llm.invoke(f"计算以下问题，只返回数字结果：\n{state['question']}").content
    logging.info(f"[计算器] 结果：{result}（{time.time()-start:.2f}s）")
    return {"tool_results": [{"content": result, "sources": ["calculator"]}]}

# 节点2c：Web搜索（模拟）
def web_search_tool(state):
    start = time.time()
    logging.info(f"[Web搜索] 模拟搜索...")
    result = f"Web搜索需要配置可访问的搜索API。当前为模拟：无法获取'{state['question']}'的实时信息。"
    logging.info(f"[Web搜索] 完成（{time.time()-start:.2f}s）")
    return {"tool_results": [{"content": result, "sources": ["web_search(模拟)"]}]}

# 节点3：生成答案
def generate_answer(state):
    start = time.time()
    logging.info(f"[生成] 生成中...")
    context = state["tool_results"][-1]["content"] if state["tool_results"] else ""
    answer = llm.invoke(f"基于以下信息回答问题。信息不足就说\"无法回答\"。\n\n信息：\n{context}\n\n问题：{state['question']}").content
    logging.info(f"[生成] 完成（{time.time()-start:.2f}s）")
    return {"answer": answer}

# 节点4：自检（PASS存记忆；3次失败明确拒答，不再硬编）
FALLBACK_ANSWER = "抱歉，基于当前知识库无法可靠回答该问题，建议联系人工客服处理。"

def reflect(state):
    start = time.time()
    logging.info(f"[自检] 检查中...")
    context = state["tool_results"][-1]["content"][:500] if state["tool_results"] else ""
    result = llm.invoke(f"检查答案是否忠于依据、是否回答了问题。\n\n问题：{state['question']}\n答案：{state['answer']}\n依据：{context}\n\n忠于依据且回答了问题返回PASS，否则返回FAIL。只返回PASS或FAIL。").content.strip()
    needs_retry = "FAIL" in result
    retry_count = state.get("retry_count", 0) + (1 if needs_retry else 0)

    degraded = False
    if needs_retry and retry_count >= 3:
        # 重试3次仍不合格：明确拒答（这本身是正确的边界行为，不缓存）
        logging.info(f"[自检] 3次未通过，降级为明确拒答")
        needs_retry = False
        degraded = True

    logging.info(f"[自检] {result}（{time.time()-start:.2f}s，重试{retry_count}次）")
    if not needs_retry and not degraded:
        save_history(state["question"], state["answer"])
    if degraded:
        return {"answer": FALLBACK_ANSWER, "needs_retry": False, "retry_count": retry_count}
    return {"needs_retry": needs_retry, "retry_count": retry_count}

# ---------- 构建状态图 ----------
workflow = StateGraph(AgentState)
workflow.add_node("classify", classify_intent)
workflow.add_node("rag", rag_tool)
workflow.add_node("calculator", calculator_tool)
workflow.add_node("web_search", web_search_tool)
workflow.add_node("generate", generate_answer)
workflow.add_node("reflect", reflect)

workflow.set_entry_point("classify")
workflow.add_conditional_edges(
    "classify",
    lambda state: "memory" if state.get("memory_hit") else state.get("tool", "rag"),
    {"memory": END, "rag": "rag", "calculator": "calculator", "web_search": "web_search"}
)
workflow.add_edge("rag", "generate")
workflow.add_edge("calculator", "generate")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", "reflect")
workflow.add_conditional_edges(
    "reflect",
    lambda state: "classify" if state.get("needs_retry") and state.get("retry_count", 0) < 3 else END,
    {"classify": "classify", END: END}
)

app = workflow.compile()

if __name__ == "__main__":
    print("\n" + "="*60 + "\n测试Agent（问题1重复问两次验证记忆）：\n")
    questions = [
        "iPhone 15的电池容量是多少？",
        "iPhone 15的电池容量是多少？",
        "退货政策是什么？几天可以退？",
        "计算 5999 减 3999 等于多少",
    ]
    for q in questions:
        print("="*60)
        result = app.invoke({"question": q, "tool": "", "tool_results": [], "answer": "", "needs_retry": False, "retry_count": 0, "memory_hit": False})
        print(f"\n>>> 最终答案：{result['answer']}")
        print(f">>> 使用工具：{result.get('tool','')}，重试：{result.get('retry_count',0)}次\n")
