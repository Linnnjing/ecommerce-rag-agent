import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

API_KEY = "9f8b164567c54051aa88b29dd4cf11f3.2Bz5nd5sxvCxCOiU"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DOCS_DIR = os.path.expanduser("~/ecommerce_rag/data/docs")
CHROMA_DIR = os.path.expanduser("~/ecommerce_rag/chroma_db")

# 1. 加载文档
print("【1】加载文档...")
loader = DirectoryLoader(DOCS_DIR, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
docs = loader.load()
print(f"    加载 {len(docs)} 篇文档")

# 2. 分块
print("【2】分块...")
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"    分块后 {len(chunks)} 块")

# 3. Embedding + 存入Chroma
print("【3】Embedding + 构建向量库（调用智谱API，稍等）...")
embeddings = OpenAIEmbeddings(model="embedding-2", openai_api_key=API_KEY, openai_api_base=BASE_URL)
batch_size = 50
vectorstore = None
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    if vectorstore is None:
        vectorstore = Chroma.from_documents(batch, embeddings, persist_directory=CHROMA_DIR)
    else:
        vectorstore.add_documents(batch)
    print(f"    已处理 {min(i+batch_size, len(chunks))}/{len(chunks)} 块")
print(f"    向量库构建完成")

# 4. LLM
llm = ChatOpenAI(model="glm-4-flash", openai_api_key=API_KEY, openai_api_base=BASE_URL, temperature=0)

# 5. LCEL构建RAG链路（langchain 1.x推荐方式，替代废弃的RetrievalQA）
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
template = """基于以下文档内容回答问题。如果文档中没有相关信息，就说"文档中没有相关信息"。

文档内容：
{context}

问题：{question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. 测试3个问题
print("\n【4】测试问答：\n")
questions = [
    "iPhone 15的电池容量是多少？",
    "退货政策是什么？几天可以退？",
    "iPhone 15和华为Mate60哪个电池大？",
]

for q in questions:
    print("=" * 60)
    print(f"问题：{q}")
    source_docs = retriever.invoke(q)
    sources = [os.path.basename(d.metadata.get('source', '')) for d in source_docs]
    answer = rag_chain.invoke(q)
    print(f"回答：{answer}")
    print(f"来源文档：{sources}")
    print()
