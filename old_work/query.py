from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# 初始化
llm = ChatOpenAI(model="deepseek-chat", temperature=0, streaming=True)

# 加载向量数据库
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vector_store = Chroma(persist_directory="./vector_db", embedding_function=embeddings)
retriever = vector_store.as_retriever()

# 定义提示词模板
prompt = ChatPromptTemplate.from_template("""
根据以下上下文回答问题：
{context}

问题：{question}
""")
# 测试检索
docs = retriever.invoke("这个框架是什么架构？")
print(f"检索到 {len(docs)} 条结果")
for d in docs:
    print(d.page_content[:100])


# LCEL管道
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    result = chain.invoke("这个框架是什么架构？")
    print(result)