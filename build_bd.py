import os
import xml.etree.ElementTree as ET
import logging
from typing import List, Optional
from langchain_community.document_loaders import (
    TextLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader, CSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from config_details import DOCS_PATH, DB_PATH, EMBEDDING_MODEL, PASSWORD

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

class SimpleXMLLoader:
    """Loader for XML files to extract text content as a Document."""
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        tree = ET.parse(self.file_path)
        root = tree.getroot()
        text = ET.tostring(root, encoding='unicode', method='text')
        return [Document(page_content=text, metadata={"source": self.file_path})]

def get_loader_for_file(path: str) -> Optional[object]:
    """Return the appropriate loader for a file based on its extension."""
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
        else:
            logging.warning(f"Unsupported file type: {path}")
            return None
    except Exception as e:
        logging.error(f"❌ Failed to load {path}: {e}")
    return None

def build_db() -> None:
    """Build the FAISS vector database from all supported documents in DOCS_PATH."""
    all_docs: List[Document] = []
    try:
        for root, _, files in os.walk(DOCS_PATH):
            for file in files:
                full_path = os.path.join(root, file)
                loader = get_loader_for_file(full_path)
                if loader:
                    try:
                        all_docs.extend(loader.load())
                    except Exception as e:
                        logging.error(f"❌ Error loading {file}: {e}")
        if not all_docs:
            logging.warning("⚠️ No documents loaded.")
            return
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(all_docs)
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(DB_PATH)
        logging.info(f"✅ Vector DB built from {len(chunks)} chunks and saved to {DB_PATH}!")
    except Exception as e:
        logging.error(f"❌ Failed to build vector DB: {e}")

if __name__ == "__main__":
    build_db()