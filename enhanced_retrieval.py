import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
import jieba

import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
API_KEY = os.environ.get("ZHIPU_API_KEY", "")
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DOCS_DIR = os.path.expanduser("~/ecommerce_rag/data/docs")
CHROMA_DIR = os.path.expanduser("~/ecommerce_rag/chroma_db")

# 加载文档+分块
print("【1】加载文档...")
loader = DirectoryLoader(DOCS_DIR, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"    {len(chunks)} 块")

# 向量检索（复用昨天的Chroma向量库）
embeddings = OpenAIEmbeddings(model="embedding-2", openai_api_key=API_KEY, openai_api_base=BASE_URL)
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# BM25检索器（关键词匹配）
print("【2】构建BM25索引...")
def chinese_preprocess(text):
    return list(jieba.cut(text))
bm25_retriever = BM25Retriever.from_documents(chunks, preprocess_func=chinese_preprocess)
bm25_retriever.k = 10

# 手动RRF混合检索（倒数排名融合，不依赖EnsembleRetriever避免import坑）
def hybrid_search(query, k=5):
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

# 测试对比：纯向量 vs 混合检索
print("\n【3】测试对比：\n")
questions = [
    "iPhone 15的电池容量是多少？",
    "退货政策是什么？几天可以退？",
    "iPhone 15和华为Mate60哪个电池大？",
]

for q in questions:
    print("=" * 60)
    print(f"问题：{q}")
    print("\n  纯向量检索top5：")
    vec_docs = vector_retriever.invoke(q)
    for d in vec_docs[:5]:
        print(f"    {os.path.basename(d.metadata.get('source',''))}")
    print("\n  混合检索(BM25+向量)top5：")
    ens_docs = hybrid_search(q, k=5)
    for d in ens_docs:
        print(f"    {os.path.basename(d.metadata.get('source',''))}")
    print()
