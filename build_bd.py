import os
import xml.etree.ElementTree as ET
import logging
from typing import List, Optional
from langchain_community.document_loaders import (
    TextLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader, CSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Handle imports for both running from root and from Logistics_Files directory
try:
    from Logistics_Files.config_details import DOCS_PATH, DB_PATH, POSTGRES_PASSWORD, MODEL_NAME, EMBEDDING_MODEL
except ImportError:
    try:
        from config_details import DOCS_PATH, DB_PATH, POSTGRES_PASSWORD, MODEL_NAME, EMBEDDING_MODEL
    except ImportError:
        raise ImportError("Could not import config_details. Make sure you're running from the correct directory.")

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
    """Build the FAISS vector database from all supported documents in DOCS_PATH using config model embeddings."""
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
        # Use optimal chunking for FAQ bot quality
        # Calculate chunk size based on embedding model dimension
        try:
            # Try to detect dimension from model name
            if 'MiniLM-L6' in EMBEDDING_MODEL:
                base_dimension = 384
            elif 'MiniLM-L12' in EMBEDDING_MODEL:
                base_dimension = 768
            elif 'all-mpnet-base-v2' in EMBEDDING_MODEL:
                base_dimension = 768
            else:
                base_dimension = 384  # Default fallback
        except:
            base_dimension = 384  # Safe fallback
        
        chunk_size = base_dimension    # Match embedding dimension for optimal performance
        chunk_overlap = int(base_dimension * 0.28)  # 28% overlap for continuity
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,  # Use character count for consistent sizing
            separators=[
                "\n\n\n",    # Major section breaks
                "\n\n",      # Paragraph breaks
                "\n",        # Line breaks
                ". ",        # Sentence endings
                "! ",        # Exclamation endings
                "? ",        # Question endings
                "; ",        # Semicolon separators
                ": ",        # Colon separators
                " - ",       # Dash separators
                " • ",       # Bullet points
                " * ",       # Asterisk separators
                " ",         # Word boundaries
                ""           # Character level (fallback)
            ]
        )
        chunks = splitter.split_documents(all_docs)
        
        # SIMPLE: Always use BGE HuggingFaceEmbeddings and from_documents
        from langchain_community.embeddings import HuggingFaceEmbeddings
        logging.info(f"Initializing {EMBEDDING_MODEL} embeddings for database build...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        logging.info(f"Creating FAISS database with {len(chunks)} chunks using {EMBEDDING_MODEL} embeddings...")
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(DB_PATH)
        logging.info(f"✅ Vector DB built from {len(chunks)} chunks using {EMBEDDING_MODEL} embeddings and saved to {DB_PATH}!")
    except Exception as e:
        logging.error(f"❌ Failed to build vector DB: {e}")

if __name__ == "__main__":
    build_db()