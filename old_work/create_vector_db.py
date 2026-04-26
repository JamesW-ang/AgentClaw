from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


def load_documents(folder_path):
    documents = []
    for file in os.listdir(folder_path):
        if file.endswith((".txt", ".md")):
            file_path = os.path.join(folder_path, file)
            with open(file_path, "r", encoding="utf-8") as f:
                from langchain_core.documents import Document
                content = f.read()
                documents.append(Document(page_content=content, metadata={"source": file}))
    return documents


def create_vector_store(docs_folder, db_path="./vector_db"):
    documents = load_documents(docs_folder)
    print(f"加载了 {len(documents)} 个文档")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"切分为 {len(chunks)} 个块")

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path,
    )
    print(f"向量数据库已保存到 {db_path}")


if __name__ == "__main__":
    create_vector_store("./docs")