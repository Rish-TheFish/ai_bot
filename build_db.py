from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, PyMuPDFLoader,
    UnstructuredWordDocumentLoader, CSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from config_details import DOCS_PATH, DB_PATH, EMBEDDING_MODEL
import os
import xml.etree.ElementTree as ET

class SimpleXMLLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        tree = ET.parse(self.file_path)
        root = tree.getroot()
        text = ET.tostring(root, encoding='unicode', method='text')
        return [Document(page_content=text, metadata={"source": self.file_path})]

def get_loader_for_file(path):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".txt":
            return TextLoader(path)
        elif ext == ".pdf":
            return PyMuPDFLoader(path)
        elif ext == ".docx":
            return UnstructuredWordDocumentLoader(path)
        elif ext == ".csv":
            return CSVLoader(path)
        elif ext == ".xml":
            return SimpleXMLLoader(path)
    except Exception as e:
        print(f"❌ Failed to load {path}: {e}")
    return None

def build_db():
    all_docs = []
    for root, _, files in os.walk(DOCS_PATH):
        for file in files:
            full_path = os.path.join(root, file)
            loader = get_loader_for_file(full_path)
            if loader:
                all_docs.extend(loader.load())

    if not all_docs:
        print("⚠️ No documents loaded.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_PATH)
    print(f"✅ Vector DB built from {len(chunks)} chunks and saved to {DB_PATH}!")

if __name__ == "__main__":
    build_db()