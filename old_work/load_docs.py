from langchain_community.document_loaders import TextLoader 
from langchain_community.document_loaders import PyMuPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
import os
from langchain_community.document_loaders import TextLoader

def load_documents(folder_path: str):
    """加载文件夹中的所有文档"""
    documents = []
    for file in os.listdir(folder_path):
        if file.endswith((".txt", ".md")):
            file_path = os.path.join(folder_path, file)
            loader = TextLoader(file_path)
            documents.extend(loader.load())
    return documents

if __name__ == "__main__":
    docs = load_documents("./docs")
    print(f"加载了 {len(docs)} 个文档")


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    return splitter.split_documents(documents)
