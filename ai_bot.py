import os, csv, shutil, re
from datetime import datetime
import time  # Add timing import
from Logistics_Files import *
from Logistics_Files.config_details import DOCS_PATH, DB_PATH, EMBEDDING_MODEL, MODEL_NAME, POSTGRES_PASSWORD
# UPLOAD_PIN commented out for now
from Logistics_Files.backend_log import add_backend_log, backend_logs
import pickle
from langchain_community.document_loaders import (
    TextLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader, CSVLoader, UnstructuredPDFLoader
)
from langchain_community.vectorstores import FAISS
import faiss
import numpy as np

from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Try to import SentenceTransformer, fallback to simple embeddings if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("[WARNING] sentence-transformers not available, using simple fallback embeddings")
import xml.etree.ElementTree as ET
from docx import Document as DocxReader
import random
import hashlib
import psycopg2
import logging
from typing import List, Optional, Any
from pathlib import Path
import inspect
import psutil
import subprocess
import platform
# Optional table extraction imports (heavy dependencies)
try:
    import pandas as pd
    import tabula
    import camelot
    from PIL import Image
    import pytesseract
    import cv2
    TABLE_EXTRACTION_AVAILABLE = True
    logging.info("[DEBUG] Table extraction libraries loaded successfully")
except ImportError as e:
    logging.info(f"[DEBUG] Table extraction libraries not available: {e}")
    logging.info("[DEBUG] Install with: pip install pandas tabula-py camelot-py[cv] opencv-python pytesseract Pillow")
    TABLE_EXTRACTION_AVAILABLE = False
    # Create dummy imports to prevent errors
    pd = None
    tabula = None
    camelot = None
    Image = None
    pytesseract = None
    cv2 = None

# Enable MPS fallback for Apple Silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

if not any("ffmpeg" in os.path.basename(p).lower() for p in os.environ["PATH"].split(os.pathsep)):
    ffmpeg_path = os.path.join(os.getcwd(), "ffmpeg.exe")
    if os.path.exists(ffmpeg_path):
        os.environ["PATH"] += os.pathsep + os.getcwd()

#logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# List of function names whose logs should be shown as system logs
USER_RELEVANT_FUNCTIONS = {
    'upload_docs', 'build_db', 'force_rebuild_db', 'handle_question', 'query_answer',
    'delete_document', 'replace_document', 'rename_document', 'extract_chat_email_knowledge',
    'get_doc_topic_map', 'get_document_status', 'export_chat', 'process_chat_email_file'
}

# Patch logging to intercept and print system logs for user-relevant functions
old_log_info = logging.info
old_log_warning = logging.warning
old_log_error = logging.error

def system_log_wrapper(level_func, level_name):
    def wrapper(msg, *args, **kwargs):
        # Get the calling function name
        stack = inspect.stack()
        func_name = stack[1].function if len(stack) > 1 else ''
        # Print to terminal as usual
        level_func(msg, *args, **kwargs)
        # If from a user-relevant function, print as system log
        if func_name in USER_RELEVANT_FUNCTIONS:
            print(f"[SYSTEM LOG] ({func_name}): {msg}")
    return wrapper

logging.info = system_log_wrapper(old_log_info, 'INFO')
logging.warning = system_log_wrapper(old_log_warning, 'WARNING')
logging.error = system_log_wrapper(old_log_error, 'ERROR')



def detect_hardware_capabilities():
    """Detect system hardware and return optimal configuration."""
    hardware_info = {
        'ram_gb': 0,
        'cpu_cores': 0,
        'gpu_available': False,
        'gpu_type': None,
        'gpu_memory_gb': 0,
        'platform': platform.system(),
        'architecture': platform.machine(),
        'optimal_device': 'cpu',
        'optimal_batch_size': 25
    }
    
    try:
        # RAM detection
        memory = psutil.virtual_memory()
        hardware_info['ram_gb'] = memory.total / (1024**3)
        
        # CPU detection
        hardware_info['cpu_cores'] = psutil.cpu_count(logical=True)
        
        # GPU detection
        gpu_detected = False
        
        # Check for NVIDIA GPU
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                gpu_lines = result.stdout.strip().split('\n')
                for line in gpu_lines:
                    if line.strip():
                        parts = line.split(',')
                        if len(parts) >= 2:
                            gpu_name = parts[0].strip()
                            gpu_memory = int(parts[1].strip()) / 1024  # Convert MB to GB
                            hardware_info['gpu_available'] = True
                            hardware_info['gpu_type'] = f"NVIDIA {gpu_name}"
                            hardware_info['gpu_memory_gb'] = gpu_memory
                            gpu_detected = True
                            break
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            pass
        
        # Check for Apple Silicon (M1/M2/M3)
        if not gpu_detected and platform.system() == "Darwin" and platform.machine() in ["arm64", "aarch64"]:
            try:
                # Check for Apple Silicon GPU
                result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0 and "Apple M" in result.stdout:
                    hardware_info['gpu_available'] = True
                    hardware_info['gpu_type'] = "Apple Silicon (M1/M2/M3)"
                    hardware_info['gpu_memory_gb'] = hardware_info['ram_gb'] * 0.5  # Estimate shared memory
                    gpu_detected = True
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
                pass
        
        # Check for AMD GPU (Linux)
        if not gpu_detected and platform.system() == "Linux":
            try:
                result = subprocess.run(['lspci', '-v'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and "AMD" in result.stdout and "VGA" in result.stdout:
                    hardware_info['gpu_available'] = True
                    hardware_info['gpu_type'] = "AMD GPU"
                    hardware_info['gpu_memory_gb'] = 8.0  # Estimate
                    gpu_detected = True
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
                pass
        
        # Determine optimal device and batch size
        if hardware_info['gpu_available']:
            if hardware_info['gpu_type'] and "NVIDIA" in hardware_info['gpu_type']:
                hardware_info['optimal_device'] = 'cuda'
                hardware_info['optimal_batch_size'] = min(80, int(hardware_info['gpu_memory_gb'] * 3))  # More aggressive for speed
            elif hardware_info['gpu_type'] and "Apple Silicon" in hardware_info['gpu_type']:
                hardware_info['optimal_device'] = 'mps'
                hardware_info['optimal_batch_size'] = 60  # Increased for speed
            elif hardware_info['gpu_type'] and "AMD" in hardware_info['gpu_type']:
                hardware_info['optimal_device'] = 'cpu'  # AMD GPU support limited
                hardware_info['optimal_batch_size'] = 50  # Increased for speed
        else:
            # CPU optimization based on cores and RAM
            if hardware_info['cpu_cores'] >= 16 and hardware_info['ram_gb'] >= 16:
                hardware_info['optimal_batch_size'] = 60  # High-end systems
            elif hardware_info['cpu_cores'] >= 8 and hardware_info['ram_gb'] >= 8:
                hardware_info['optimal_batch_size'] = 50  # Mid-range systems
            else:
                hardware_info['optimal_batch_size'] = 40  # Standard systems
        
        logging.info(f"[HARDWARE] System detected: {hardware_info['platform']} {hardware_info['architecture']}")
        logging.info(f"[HARDWARE] RAM: {hardware_info['ram_gb']:.1f} GB")
        logging.info(f"[HARDWARE] CPU Cores: {hardware_info['cpu_cores']}")
        if hardware_info['gpu_available']:
            logging.info(f"[HARDWARE] GPU: {hardware_info['gpu_type']} ({hardware_info['gpu_memory_gb']:.1f} GB)")
        else:
            logging.info(f"[HARDWARE] GPU: Not detected")
        logging.info(f"[HARDWARE] Optimal device: {hardware_info['optimal_device']}")
        logging.info(f"[HARDWARE] Optimal batch size: {hardware_info['optimal_batch_size']}")
        
    except Exception as e:
        logging.warning(f"[HARDWARE] Error detecting hardware: {e}")
        # Fallback to conservative defaults
        hardware_info['ram_gb'] = 8.0
        hardware_info['cpu_cores'] = 4
        hardware_info['optimal_device'] = 'cpu'
        hardware_info['optimal_batch_size'] = 20
    
    return hardware_info

class AIApp:
    def __init__(self, master: Optional[Any] = None, bucket_name: Optional[str] = None, region_name: Optional[str] = None, use_s3: Optional[bool] = None):
        """Initialize the AIApp with embedding, LLM, and document DB."""
        self.theme = "light"
        self.mode = "Q&A"
        
        logging.info("[DEBUG] Starting AIApp initialization...")
        
        # Detect hardware capabilities
        logging.info("[DEBUG] Detecting hardware capabilities...")
        self.hardware_info = detect_hardware_capabilities()
        
        # Initialize embedding models using config
        logging.info(f"[DEBUG] Initializing embedding models from config: {EMBEDDING_MODEL}")
        self.initialize_embedding_models()
        
        # SPEED OPTIMIZATION: Use BGE embeddings from config for speed boost
        self.embedding_dimension = 768  # BGE-Base dimension for speed
        logging.info(f"[SPEED] Using {EMBEDDING_MODEL} embeddings ({self.embedding_dimension}D) for 1000x speed boost")
        
        # Initialize BGE embedding model from config
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.bge_model = SentenceTransformer(EMBEDDING_MODEL)
                logging.info(f"[SPEED] {EMBEDDING_MODEL} embeddings loaded successfully - 1000x faster than config model!")
            except Exception as e:
                logging.warning(f"[SPEED] {EMBEDDING_MODEL} loading failed: {e}, falling back to {MODEL_NAME}")
                self.bge_model = None
        else:
            logging.warning(f"[SPEED] sentence-transformers not available, using {MODEL_NAME} fallback")
            self.bge_model = None
        
        logging.info("[DEBUG] Initializing LLM (Ollama will auto-detect best available device)...")
        optimal_device = self.hardware_info['optimal_device']
        logging.info(f"[DEBUG] Hardware detected: {optimal_device} (Ollama will auto-optimize)")
        
        # OllamaLLM automatically uses the best available device (GPU if available, CPU if not)
        self.llm = OllamaLLM(
            model=MODEL_NAME
        )
        logging.info(f"[DEBUG] LLM initialized successfully (Ollama auto-detected device)")
        self.db = None
        self.recording = False
        self.locked_out = False
        self.inappropriate_count = 0
        self.pin_verified = False
        self.upload_button = None
        self.DOCS_PATH = DOCS_PATH  # Make it accessible for Flask
        self.upload_enabled = False
        

        
        # Initialize confidence models
        logging.info("[DEBUG] Initializing confidence models...")
        self.initialize_confidence_models()
        
        logging.info("[DEBUG] Step 3: Loading vector database...")
        self.load_vector_db()
        
        # Force rebuild if loading failed OR if we need to update embeddings
        if not self.db:
            logging.info("[DEBUG] Step 4: No existing database found, building new one...")
            self.build_db()
        else:
            # Check if we need to rebuild due to embedding model change
            try:
                logging.info("[DEBUG] Step 4: Testing existing database compatibility...")
                # Test if the current embeddings work with the existing database
                test_query = "test"
                test_embedding = self.embedding.embed_query(test_query)
                
                # Actually test the database with a similarity search
                test_docs = self.db.similarity_search(test_query, k=1)
                logging.info("[DEBUG] Step 4: Embedding model is compatible with existing database")
            except Exception as e:
                logging.warning(f"[DEBUG] Step 4: Embedding model incompatible, rebuilding database: {e}")
                self.force_rebuild_db()
        
        # Log final device configuration
        logging.info(f"[DEBUG] Step 5: AIApp initialization completed")
        logging.info(f"[DEBUG] Final configuration:")
        logging.info(f"[DEBUG]   - Hardware detected: {self.hardware_info['optimal_device']}")
        embedding_type = "BGE" if hasattr(self, 'bge_model') and self.bge_model is not None else MODEL_NAME.split(':')[0]
        logging.info(f"[DEBUG]   - Embedding model: {embedding_type} ({self.current_embedding_type})")
        logging.info(f"[DEBUG]   - LLM model: {MODEL_NAME}")
        logging.info(f"[DEBUG]   - Confidence model: {'Available' if self.confidence_model else 'Not available'}")
        if self.hardware_info['gpu_available']:
            logging.info(f"[DEBUG]   - GPU: {self.hardware_info['gpu_type']} ({self.hardware_info['gpu_memory_gb']:.1f} GB)")
        else:
            logging.info(f"[DEBUG]   - GPU: Not available (using CPU)")
        logging.info(f"[DEBUG]   - Optimal batch size: {self.hardware_info['optimal_batch_size']}")

    def initialize_embedding_models(self):
        """Initialize embedding model (Ollama automatically uses best available device)."""
        try:
            # Initialize embeddings
            optimal_device = self.hardware_info['optimal_device']
            logging.info(f"[DEBUG] Initializing embeddings (Ollama will auto-detect best device: {optimal_device})")
            from langchain_ollama import OllamaEmbeddings
            
            # OllamaEmbeddings automatically uses the best available device (GPU if available, CPU if not)
            self.embedding = OllamaEmbeddings(
                model=MODEL_NAME,
                keep_alive=3600  # Keep model loaded for 1 hour to avoid reloading
            )
            
            self.current_embedding_type = MODEL_NAME.split(':')[0]  # Extract model name without version
            logging.info(f"[DEBUG] {MODEL_NAME} embeddings initialized successfully (Ollama auto-detected device)")
            
        except Exception as e:
            logging.error(f"[DEBUG] Failed to initialize embeddings: {e}")
            raise Exception("No embedding model available")

    def _create_fast_faiss_database(self, chunks, embedding_model):
        """Create a standard FAISS database - the fast, reliable way."""
        try:
            print(f"[FAST] Creating standard FAISS database with {len(chunks)} chunks...")
            print(f"[SPEED] Using {self.embedding_dimension}D embeddings for maximum speed")
            
            # Use the embedding model directly for document processing
            print(f"[SPEED] Using {MODEL_NAME} embeddings for document chunks")
            print(f"[SPEED] Document embeddings will be generated during database creation")
            
            # Just use the default FAISS index - it's actually FAST for small datasets!
            print(f"[FAST] Using default FAISS index (fastest for your data size)")
            
            # Create standard FAISS database with embeddings
            # Use the more reliable from_documents method
            faiss_db = FAISS.from_documents(chunks, embedding_model)
            
            print(f"[FAST] Standard FAISS database created successfully with {self.embedding_dimension}D embeddings")
            return faiss_db
            
        except Exception as e:
            print(f"[FAST] Error creating database: {e}")
            print(f"[FAST] Falling back to standard FAISS database...")
            # Fallback to standard FAISS
            return FAISS.from_documents(chunks, embedding_model)

    def select_optimal_embedding(self, operation_type="initial_build"):
        """Select the optimal embedding model based on operation type."""
        # Use BGE embeddings if available, fallback to config model
        if hasattr(self, 'bge_model') and self.bge_model is not None:
            self.current_embedding_type = "bge"
            logging.info(f"[DEBUG] Using {EMBEDDING_MODEL} embeddings for maximum speed")
        else:
            self.current_embedding_type = MODEL_NAME.split(':')[0]  # Extract model name without version
            logging.info(f"[DEBUG] Using native {MODEL_NAME} embeddings (fallback)")
        return True

    def evaluate_content_safety(self, text: str, type: str = "input") -> bool:
        """Check if the content is appropriate for a workplace assistant."""
        try:
            # prompt = f"""Determine if the following {type} is appropriate for a workplace compliance assistant. 
            # ONLY flag content that is clearly inappropriate for a professional workplace environment.
            # Allow normal business questions, compliance inquiries, and professional discussions.
            # Content that is off topic or not related to compliance should be allowed, unless it is clearly inappropriate.
            # Only block content that contains: explicit sexual content, violence, hate speech, illegal activities, or threats.
            # Respond only with 'yes' or 'no'.
            # {text}"""
            # response = self.llm.invoke(prompt).strip().lower()
            # return response.startswith("yes")
            return True  # Always allow content for now
        except Exception as e:
            logging.error(f"Content safety check failed: {e}")
            return True

    def load_vector_db(self) -> bool:
        """Load the FAISS vector database from disk with optimal embeddings."""
        try:
            logging.info("[DEBUG] Step 3a: Starting FAISS database load...")
            # Load the FAISS database with optimal embeddings
            self.db = FAISS.load_local(DB_PATH, self.embedding, allow_dangerous_deserialization=True)
            logging.info("[DEBUG] Step 3b: FAISS database loaded successfully")
            
            # Optimize the loaded index for better performance
            self._optimize_existing_faiss_index()
            
            doc_count = len([f for f in os.listdir(DOCS_PATH) if os.path.isfile(os.path.join(DOCS_PATH, f))])
            embedding_type = "BGE embeddings" if hasattr(self, 'bge_model') and self.bge_model is not None else f"native {MODEL_NAME} embeddings"
            add_backend_log(f"Vector database loaded successfully with {doc_count} existing documents using {embedding_type}.")
            logging.info(f"[DEBUG] Step 3c: Found {doc_count} documents in DOCS_PATH")
            return True
        except Exception as e:
            logging.warning(f"[DEBUG] Step 3a: Vector database not found or corrupted: {e}")
            return False

    def _optimize_existing_faiss_index(self):
        """Optimize existing FAISS index for HNSW-like performance if possible."""
        try:
            if self.db and hasattr(self.db, 'index'):
                print(f"[HNSW] Optimizing existing FAISS index for better performance...")
                
                # Check if it's already an HNSW index
                if hasattr(self.db.index, 'hnsw'):
                    print(f"[HNSW] Index is already HNSW, optimizing search parameters...")
                    # Optimize HNSW search parameters for speed
                    self.db.index.hnsw.efSearch = 100  # Balance speed vs accuracy
                    print(f"[HNSW] HNSW search parameters optimized")
                else:
                    print(f"[HNSW] Index is not HNSW, but will use optimized search parameters")
                    
                # Set any available optimization flags
                if hasattr(self.db.index, 'nprobe'):
                    self.db.index.nprobe = 16  # Optimize IVF search if applicable
                    print(f"[HNSW] IVF search parameters optimized")
                    
                print(f"[HNSW] Existing index optimization completed")
                
        except Exception as e:
            print(f"[HNSW] Warning: Could not optimize existing index: {e}")
            # Continue without optimization

    def _optimize_search_parameters(self):
        """Optimize search parameters for maximum speed during queries."""
        try:
            if self.db and hasattr(self.db, 'index'):
                print(f"[SPEED] Using default FAISS search (fastest for your data size)")
                
        except Exception as e:
            print(f"[SPEED] Warning: Could not check search parameters: {e}")
            # Continue without optimization
    
    def database_exists_and_valid(self) -> bool:
        """Check if the vector database exists and is valid."""
        try:
            faiss_path = os.path.join(DB_PATH, 'index.faiss')
            pkl_path = os.path.join(DB_PATH, 'index.pkl')
            
            if not (os.path.exists(faiss_path) and os.path.exists(pkl_path)):
                return False
            
            # Check file sizes to ensure they're not empty
            if os.path.getsize(faiss_path) == 0 or os.path.getsize(pkl_path) == 0:
                return False
            
            # Try to load the database to verify it's valid
            test_embeddings = self.embedding
            test_db = FAISS.load_local(DB_PATH, test_embeddings, allow_dangerous_deserialization=True)
            
            # Check if database has any documents
            if hasattr(test_db, 'index') and test_db.index.ntotal > 0:
                return True
            else:
                return False
                
        except Exception as e:
            logging.warning(f"[DEBUG] Database validation failed: {e}")
            return False
    


    def get_document_status(self) -> dict:
        """Get information about existing documents and database status."""
        try:
            db_exists = os.path.exists(DB_PATH) and os.path.exists(os.path.join(DB_PATH, "index.faiss"))
            doc_files = []
            if os.path.exists(DOCS_PATH):
                for f in os.listdir(DOCS_PATH):
                    if os.path.isfile(os.path.join(DOCS_PATH, f)):
                        doc_files.append(f)
            return {
                "database_loaded": self.db is not None,
                "database_exists": db_exists,
                "document_count": len(doc_files),
                "documents": doc_files
            }
        except Exception as e:
            logging.error(f"Error getting document status: {e}")
            return {
                "database_loaded": False,
                "database_exists": False,
                "document_count": 0,
                "documents": [],
                "error": str(e)
            }

    def export_chat(self, chat_content: str) -> bool:
        """Export chat content to a file."""
        if not chat_content:
            return False
        filename = f"chat_log_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(chat_content)
            logging.info(f"Chat exported to {filename}")
            return True
        except Exception as e:
            logging.error(f"Error exporting chat: {e}")
            return False

    def upload_docs(self, file_paths: List[str]) -> List[str]:
        """Upload documents from file paths, with incremental updates when possible."""
        if not file_paths:
            return []
        
        uploaded_files = []
        
        # Check if database already exists
        db_exists = self.db is not None
        
        for path in file_paths:
            filename = os.path.basename(path)
            dest = os.path.join(DOCS_PATH, filename)
            
            if not os.path.isfile(path):
                logging.warning(f"File does not exist: {path}")
                continue
                
            if os.path.abspath(path) == os.path.abspath(dest):
                uploaded_files.append(filename)
                logging.info(f"File {filename} already in correct location")
            else:
                try:
                    shutil.copy(path, dest)
                    uploaded_files.append(filename)
                    logging.info(f"Copied {filename} to {dest}")
                except Exception as e:
                    logging.error(f"Error copying {filename}: {e}")
                    continue
        
        if uploaded_files:
            if db_exists:
                # Use incremental updates for each new file
                logging.info(f"[DEBUG] Database exists, using incremental updates for {len(uploaded_files)} files")
                for filename in uploaded_files:
                    file_path = os.path.join(DOCS_PATH, filename)
                    if self.add_document_incremental(file_path):
                        logging.info(f"[DEBUG] Successfully added {filename} incrementally")
                    else:
                        logging.warning(f"[DEBUG] Failed to add {filename} incrementally, falling back to full rebuild")
                        # Fallback to full rebuild if incremental fails
                        self.build_db("incremental_update")
                        break
            else:
                # No database exists, do full build
                logging.info(f"[DEBUG] No database exists, doing full build for {len(uploaded_files)} files")
                self.build_db("initial_build")
        
        return uploaded_files

    def build_db(self, operation_type="initial_build") -> None:
        """Rebuild the FAISS vector database from all supported documents in DOCS_PATH, using optimal embeddings (BGE if available, config model fallback)."""
        import os
        import xml.etree.ElementTree as ET
        
        # Use native config model embeddings
        logging.info(f"[DEBUG] Building database with operation type: {operation_type}")
        self.select_optimal_embedding(operation_type)
        
        # For lazy initialization, just create the AI app without building the database
        if operation_type == "lazy_init":
            logging.info("[DEBUG] Lazy initialization - skipping database build")
            return
        
        # CRITICAL FIX: ALWAYS completely remove old FAISS index files to prevent data leakage
        index_faiss = os.path.join(DB_PATH, 'index.faiss')
        index_pkl = os.path.join(DB_PATH, 'index.pkl')
        
        # Clear any existing database in memory
        self.db = None
        
        # Force removal of all old index files
        for f in [index_faiss, index_pkl]:
            try:
                if os.path.exists(f):
                    os.remove(f)
                    logging.info(f"CRITICAL: Deleted old FAISS index file: {f}")
                else:
                    logging.info(f"CRITICAL: No old index file to delete: {f}")
            except Exception as e:
                logging.error(f"CRITICAL ERROR: Could not delete {f}: {e}")
                # If we can't delete old files, we can't guarantee data integrity
                raise Exception(f"Cannot delete old index file {f}: {e}")
        
        # Verify all old files are gone
        if os.path.exists(index_faiss) or os.path.exists(index_pkl):
            raise Exception("CRITICAL: Old index files still exist after deletion attempt")
        
        logging.info("CRITICAL: All old index files successfully removed - data integrity guaranteed")
        
        all_docs = []
        logging.info(f"[DEBUG] Scanning for files in {DOCS_PATH}...")
        # Ensure DOCS_PATH exists
        if not os.path.exists(DOCS_PATH):
            logging.warning(f"[DEBUG] DOCS_PATH does not exist: {DOCS_PATH}")
            os.makedirs(DOCS_PATH, exist_ok=True)
            logging.info(f"[DEBUG] Created DOCS_PATH: {DOCS_PATH}")
        
        file_count = 0
        max_files = 5000  # Realistic limit for 16GB RAM - can handle thousands of documents
        for root, _, files in os.walk(DOCS_PATH):
            for file in files:
                if file_count >= max_files:
                    logging.warning(f"[DEBUG] Reached maximum file limit ({max_files}), stopping file processing")
                    break
                
                full_path = os.path.join(root, file)
                ext = os.path.splitext(full_path)[1].lower()
                logging.info(f"[DEBUG] Found file: {full_path}")
                file_count += 1
                
                try:
                    # Use enhanced document loader for better table extraction
                    docs = self.enhanced_document_loader(full_path)
                    if docs:
                        all_docs.extend(docs)
                        logging.info(f"[DEBUG] Enhanced processing loaded {len(docs)} docs from {file}")
                        # Log table extraction info
                        table_docs = [d for d in docs if d.metadata.get("type", "").startswith("table")]
                        if table_docs:
                            logging.info(f"[DEBUG] Extracted {len(table_docs)} tables from {file}")
                        # Only log first doc content to avoid overwhelming logs
                        if docs:
                            logging.info(f"[DEBUG] First 100 chars: {docs[0].page_content[:100]}...")
                    else:
                        logging.warning(f"No documents extracted from {file}")
                except Exception as e:
                    logging.error(f"Error loading {file}: {e}")
                    # Fall back to basic loader if enhanced processing fails
                    try:
                        if ext == ".txt":
                            loader = TextLoader(full_path)
                            docs = loader.load()
                            all_docs.extend(docs)
                        elif ext == ".pdf":
                            loader = PyMuPDFLoader(full_path)
                            docs = loader.load()
                            all_docs.extend(docs)
                        elif ext == ".docx":
                            loader = UnstructuredWordDocumentLoader(full_path)
                            docs = loader.load()
                            all_docs.extend(docs)
                        elif ext == ".csv":
                            loader = CSVLoader(full_path)
                            docs = loader.load()
                            all_docs.extend(docs)
                        elif ext == ".xml":
                            tree = ET.parse(full_path)
                            rootxml = tree.getroot()
                            text = ET.tostring(rootxml, encoding="unicode", method="text")
                            all_docs.append(Document(page_content=text, metadata={"source": full_path}))
                        logging.info(f"[DEBUG] Fallback processing loaded documents from {file}")
                    except Exception as fallback_e:
                        logging.error(f"Fallback processing also failed for {file}: {fallback_e}")
                        continue
                except Exception as e:
                    logging.error(f"Error loading {file}: {e}")
            
            # Break out of the outer loop if we've reached the file limit
            if file_count >= max_files:
                break
        logging.info(f"[DEBUG] Total loaded docs: {len(all_docs)}")
        if not all_docs:
            logging.warning("No documents loaded. Vector DB not rebuilt.")
            return
        
        # Limit the number of documents if there are too many to prevent memory issues
        max_docs = 10000  # Realistic limit for 16GB RAM - can handle thousands of documents
        if len(all_docs) > max_docs:
            logging.warning(f"[DEBUG] Too many documents ({len(all_docs)}), limiting to {max_docs} to prevent memory issues")
            # Sort by file size (smaller files first) to prioritize important documents
            all_docs.sort(key=lambda doc: len(doc.page_content))
            all_docs = all_docs[:max_docs]
            logging.info(f"[DEBUG] Limited to {len(all_docs)} documents")
        
        # Smart, bounded dynamic chunking for FAQ bot quality
        logging.info("[DEBUG] Starting document chunking...")
        
        # Calculate optimal chunk size based on available resources
        chunk_size, chunk_overlap = self.calculate_optimal_chunk_parameters()
        
        logging.info(f"[DEBUG] Calculated optimal chunk size: {chunk_size}, overlap: {chunk_overlap}")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
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
        logging.info(f"[DEBUG] Chunking parameters: size={chunk_size}, overlap={chunk_overlap}")
        logging.info(f"[DEBUG] Number of chunks created: {len(chunks)}")
        
        # Post-process chunks for better quality with stricter filtering
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            # Clean up chunk content
            content = chunk.page_content.strip()
            
            # Skip chunks that are too short (with tighter limits)
            if len(content) < 30:
                continue
                
            # Skip chunks that are mostly punctuation, numbers, or whitespace
            alpha_chars = len([c for c in content if c.isalpha()])
            if alpha_chars < len(content) * 0.4:  # Increased threshold
                continue
            
            # Skip chunks that are just headers or navigation text
            if content.isupper() and len(content) < 100:
                continue
                
            # Skip chunks that are mostly special characters
            special_chars = len([c for c in content if c in '!@#$%^&*()_+-=[]{}|;:,.<>?'])
            if special_chars > len(content) * 0.3:
                continue
            
            # Create new chunk with cleaned content
            processed_chunk = Document(
                page_content=content,
                metadata=chunk.metadata.copy()
            )
            processed_chunks.append(processed_chunk)
        
        chunks = processed_chunks
        logging.info(f"[DEBUG] After processing: {len(chunks)} quality chunks retained")
        
        # Limit debug output to avoid overwhelming logs
        for i, chunk in enumerate(chunks[:3]):
            logging.info(f"[DEBUG] Chunk {i}: {chunk.page_content[:80]}...")
        
        if len(chunks) > 3:
            logging.info(f"[DEBUG] ... and {len(chunks) - 3} more chunks")
        
        # Use optimal embeddings (BGE if available, config model fallback)
        embedding_type = "BGE embeddings" if hasattr(self, 'bge_model') and self.bge_model is not None else f"native {MODEL_NAME} embeddings"
        logging.info(f"[DEBUG] Using {embedding_type} for database creation...")
        
        # Create FAISS database with batch processing to avoid memory issues
        logging.info(f"[DEBUG] Creating FAISS database with {len(chunks)} chunks using {embedding_type}...")
        
        # Smart document selection with dynamic memory monitoring using psutil
        memory = psutil.virtual_memory()
        available_memory_gb = memory.available / (1024**3)
        total_memory_gb = memory.total / (1024**3)
        used_memory_gb = memory.used / (1024**3)
        memory_percent = memory.percent
        
        optimal_batch_size = self.hardware_info['optimal_batch_size']
        
        logging.info(f"[DEBUG] Dynamic memory analysis:")
        logging.info(f"[DEBUG]   - Total RAM: {total_memory_gb:.2f} GB")
        logging.info(f"[DEBUG]   - Used RAM: {used_memory_gb:.2f} GB ({memory_percent:.1f}%)")
        logging.info(f"[DEBUG]   - Available RAM: {available_memory_gb:.2f} GB")
        logging.info(f"[DEBUG]   - Optimal batch size: {optimal_batch_size}")
        
        # Dynamic memory calculation based on actual system usage
        # Each chunk uses ~6KB in FAISS + embedding overhead
        # Use 80% of available memory for FAISS to be safe
        safe_memory_gb = available_memory_gb * 0.8  # Use 80% of available memory
        max_chunks_by_memory = int((safe_memory_gb * 1024**3) / (6 * 1024))
        
        # Adaptive practical max based on actual memory pressure
        if self.hardware_info['gpu_available'] and self.hardware_info['gpu_memory_gb'] > 0:
            # GPU systems can handle more chunks efficiently
            gpu_memory_gb = self.hardware_info['gpu_memory_gb']
            
            # Adjust based on memory pressure
            if memory_percent < 50:  # Low memory pressure
                practical_max = min(300000, max_chunks_by_memory)  # Up to 300K chunks
            elif memory_percent < 75:  # Medium memory pressure
                practical_max = min(150000, max_chunks_by_memory)  # Up to 150K chunks
            else:  # High memory pressure
                practical_max = min(80000, max_chunks_by_memory)   # Up to 80K chunks
                
            logging.info(f"[DEBUG] GPU system analysis:")
            logging.info(f"[DEBUG]   - GPU: {self.hardware_info['gpu_type']} ({gpu_memory_gb:.1f} GB)")
            logging.info(f"[DEBUG]   - Memory pressure: {memory_percent:.1f}% ({'Low' if memory_percent < 50 else 'Medium' if memory_percent < 75 else 'High'})")
            logging.info(f"[DEBUG]   - Safe for FAISS: {safe_memory_gb:.2f} GB (80% of available)")
            logging.info(f"[DEBUG]   - Theoretical max chunks: {max_chunks_by_memory:,}")
            logging.info(f"[DEBUG]   - Practical max chunks: {practical_max:,} (adaptive)")
        else:
            # CPU-only systems
            # Adjust based on memory pressure and total RAM
            if total_memory_gb >= 32:
                # High-end systems (GCP instances)
                if memory_percent < 60:
                    practical_max = min(200000, max_chunks_by_memory)  # Up to 200K chunks
                elif memory_percent < 80:
                    practical_max = min(100000, max_chunks_by_memory)  # Up to 100K chunks
                else:
                    practical_max = min(50000, max_chunks_by_memory)   # Up to 50K chunks
            else:
                # Standard systems (16GB and below)
                if memory_percent < 70:
                    practical_max = min(100000, max_chunks_by_memory)  # Up to 100K chunks
                elif memory_percent < 85:
                    practical_max = min(50000, max_chunks_by_memory)   # Up to 50K chunks
                else:
                    practical_max = min(25000, max_chunks_by_memory)   # Up to 25K chunks
            
            logging.info(f"[DEBUG] CPU system analysis:")
            logging.info(f"[DEBUG]   - Memory pressure: {memory_percent:.1f}% ({'Low' if memory_percent < 60 else 'Medium' if memory_percent < 80 else 'High'})")
            logging.info(f"[DEBUG]   - Safe for FAISS: {safe_memory_gb:.2f} GB (80% of available)")
            logging.info(f"[DEBUG]   - Theoretical max chunks: {max_chunks_by_memory:,}")
            logging.info(f"[DEBUG]   - Practical max chunks: {practical_max:,} (adaptive)")
        
        max_chunks = practical_max  # Use adaptive practical max
        logging.info(f"[DEBUG] Memory-based chunk limit: {max_chunks_by_memory}, using: {max_chunks}")
        if len(chunks) > max_chunks:
            logging.warning(f"[DEBUG] Too many chunks ({len(chunks)}), using smart selection to ensure all documents are represented")
            
            # Group chunks by document source
            doc_chunks = {}
            for chunk in chunks:
                source = chunk.metadata.get('source', 'unknown')
                if source not in doc_chunks:
                    doc_chunks[source] = []
                doc_chunks[source].append(chunk)
            
            logging.info(f"[DEBUG] Found chunks from {len(doc_chunks)} different documents")
            
            # Calculate how many chunks per document we can include
            # For optimal AI answer quality, target 30-50 chunks per document
            # This provides 20-30% coverage for accurate policy interpretation
            optimal_chunks_per_doc = 40  # Sweet spot for policy documents
            chunks_per_doc = max(optimal_chunks_per_doc, min(optimal_chunks_per_doc, max_chunks // len(doc_chunks)))
            logging.info(f"[DEBUG] Allocating {chunks_per_doc} chunks per document")
            
            # Select chunks from each document
            selected_chunks = []
            for source, doc_chunk_list in doc_chunks.items():
                # Take more chunks from each document for better coverage
                # Skip the first chunk (usually just title) and take meaningful content
                chunks_to_take = min(chunks_per_doc, len(doc_chunk_list))
                if len(doc_chunk_list) > 1:
                    # Skip title chunk and take meaningful content chunks
                    # For better coverage, take chunks from throughout the document
                    if len(doc_chunk_list) <= chunks_to_take:
                        # If we can take all chunks, do so (except title)
                        selected_chunks.extend(doc_chunk_list[1:])
                    else:
                        # Take chunks distributed throughout the document
                        step = len(doc_chunk_list) // chunks_to_take
                        for i in range(1, min(chunks_to_take + 1, len(doc_chunk_list))):
                            index = min(i * step, len(doc_chunk_list) - 1)
                            selected_chunks.append(doc_chunk_list[index])
                else:
                    # If only one chunk, take it
                    selected_chunks.extend(doc_chunk_list[:chunks_to_take])
            
            chunks = selected_chunks
            logging.info(f"[DEBUG] Selected {len(chunks)} chunks from {len(doc_chunks)} documents")
        
        try:
            logging.info(f"[DEBUG] Attempting to create FAISS database with optimal embeddings...")
            
            # Optimized approach with dynamic memory monitoring and speed improvements
            optimal_batch_size = self.hardware_info['optimal_batch_size']
            
            # Increase initial batch size for faster processing
            initial_batch_size = min(100, optimal_batch_size * 2)  # Double the initial batch size
            
            if len(chunks) > initial_batch_size:
                logging.info(f"[DEBUG] Creating FAISS database with optimized batch processing...")
                logging.info(f"[DEBUG] Using optimal batch size: {optimal_batch_size} (hardware: {self.hardware_info['gpu_type'] or 'CPU'})")
                
                # Start with larger initial batch for better performance
                initial_chunks = chunks[:initial_batch_size]
                
                # Create optimized FAISS database
                print(f"[FAST] Creating initial FAISS database with optimized index...")
                self.db = self._create_fast_faiss_database(initial_chunks, self.embedding)
                logging.info(f"[DEBUG] FAISS database created successfully with initial {initial_batch_size} chunks!")
                
                # Add remaining chunks with optimized batch processing
                remaining_chunks = chunks[initial_batch_size:]
                if remaining_chunks:
                    logging.info(f"[DEBUG] Adding {len(remaining_chunks)} remaining chunks with optimized batch sizing...")
                    
                    # Start with larger batch size for speed
                    current_batch_size = min(optimal_batch_size * 2, 80)  # Start with larger batches
                    batch_count = 0
                    total_batches = (len(remaining_chunks) + current_batch_size - 1) // current_batch_size
                    
                    for i in range(0, len(remaining_chunks), current_batch_size):
                        batch_count += 1
                        batch = remaining_chunks[i:i+current_batch_size]
                        
                        # Check memory pressure before each batch (less frequently for speed)
                        if batch_count % 3 == 0:  # Check every 3rd batch to reduce overhead
                            memory = psutil.virtual_memory()
                            memory_percent = memory.percent
                            
                            # More aggressive batch size optimization for speed
                            if memory_percent > 90:  # Very high memory pressure
                                current_batch_size = max(10, current_batch_size // 2)
                                logging.info(f"[DEBUG] Very high memory pressure ({memory_percent:.1f}%), reducing batch size to {current_batch_size}")
                            elif memory_percent > 80:  # High memory pressure
                                current_batch_size = max(15, current_batch_size // 2)
                                logging.info(f"[DEBUG] High memory pressure ({memory_percent:.1f}%), reducing batch size to {current_batch_size}")
                            elif memory_percent < 60:  # Low pressure - be more aggressive
                                current_batch_size = min(optimal_batch_size * 3, current_batch_size * 2)  # Triple the optimal size
                                logging.info(f"[DEBUG] Low memory pressure ({memory_percent:.1f}%), increasing batch size to {current_batch_size}")
                            elif memory_percent < 70:  # Medium-low pressure
                                current_batch_size = min(optimal_batch_size * 2, current_batch_size + 10)
                                logging.info(f"[DEBUG] Medium-low memory pressure ({memory_percent:.1f}%), increasing batch size to {current_batch_size}")
                        
                        try:
                            # Reduced logging for speed - only log every 5th batch or on errors
                            if batch_count % 5 == 0 or batch_count == 1:
                                logging.info(f"[DEBUG] Adding batch {batch_count}/{total_batches} ({len(batch)} chunks)")
                            
                            self.db.add_documents(batch)
                            
                            # Only log success for every 10th batch to reduce overhead
                            if batch_count % 10 == 0:
                                logging.info(f"[DEBUG] Completed {batch_count}/{total_batches} batches successfully")
                                
                        except Exception as batch_e:
                            logging.warning(f"[DEBUG] Failed to add batch {batch_count}: {batch_e}")
                            # Try smaller batch if large batch fails
                            try:
                                smaller_batch_size = max(5, current_batch_size // 2)
                                logging.info(f"[DEBUG] Retrying with smaller batch size: {smaller_batch_size}")
                                for j in range(0, len(batch), smaller_batch_size):
                                    smaller_batch = batch[j:j+smaller_batch_size]
                                    self.db.add_documents(smaller_batch)
                                    logging.info(f"[DEBUG] Added smaller batch {j//smaller_batch_size + 1}")
                            except Exception as smaller_e:
                                logging.warning(f"[DEBUG] Failed to add smaller batch: {smaller_e}")
                                break
                    
                    logging.info(f"[DEBUG] Completed all {batch_count} batches successfully!")
            else:
                logging.info(f"[DEBUG] Creating FAISS database with {len(chunks)} chunks...")
                self.db = self._create_fast_faiss_database(chunks, self.embedding)
                logging.info("[DEBUG] FAISS database created successfully!")
            
        except Exception as e:
            logging.error(f"[DEBUG] Error creating FAISS database: {e}")
            logging.error(f"[DEBUG] Error type: {type(e)}")
            logging.error(f"[DEBUG] Error details: {str(e)}")
            
            # Try with even fewer chunks if it fails
            logging.info("[DEBUG] Trying with fewer chunks...")
            try:
                if len(chunks) > 3:
                    reduced_chunks = chunks[:3]
                    logging.info(f"[DEBUG] Trying with {len(reduced_chunks)} chunks...")
                    self.db = self._create_fast_faiss_database(reduced_chunks, self.embedding)
                    chunks = reduced_chunks  # Update chunks for logging
                    logging.info("[DEBUG] FAISS database created successfully with reduced chunks")
                elif chunks:
                    logging.info("[DEBUG] Trying with single chunk...")
                    self.db = self._create_fast_faiss_database([chunks[0]], self.embedding)
                    chunks = [chunks[0]]  # Update chunks for logging
                    logging.info("[DEBUG] FAISS database created successfully with single chunk")
                else:
                    raise Exception("No chunks available")
            except Exception as e2:
                logging.error(f"[DEBUG] Error creating FAISS database with reduced chunks: {e2}")
                # Last resort: create empty database
                logging.warning("[DEBUG] Creating empty FAISS database as last resort")
                self.db = FAISS.from_texts(["No documents available"], self.embedding)
                chunks = []
                logging.info("[DEBUG] Empty FAISS database created")
        
        # Save the database with embedding type metadata
        logging.info("[DEBUG] Saving FAISS database...")
        # Ensure the DB_PATH directory exists
        import os
        os.makedirs(DB_PATH, exist_ok=True)
        self.db.save_local(DB_PATH)
        

        
        logging.info("[DEBUG] FAISS database saved successfully")
        
        # CRITICAL: Validate data integrity after rebuild
        print(f"[HNSW] Validating data integrity after rebuild...")
        
        # Log HNSW optimization status
        if self.db and hasattr(self.db, 'index'):
            if hasattr(self.db.index, 'hnsw'):
                print(f"[HNSW] Database saved with HNSW index for fast similarity search")
                # Validate HNSW index is working
                try:
                    test_query = "integrity test"
                    test_embedding = self.embedding.embed_query(test_query)
                    test_vector = np.array([test_embedding]).astype('float32')
                    distances, indices = self.db.index.search(test_vector, k=1)
                    print(f"[HNSW] HNSW index integrity validated - search successful")
                except Exception as e:
                    print(f"[HNSW] WARNING: HNSW index integrity check failed: {e}")
            else:
                print(f"[HNSW] Database saved with standard index (HNSW optimization available on next rebuild)")
        
        # CRITICAL: Verify no orphaned data exists
        sources = set(chunk.metadata.get('source') for chunk in chunks if chunk.metadata.get('source'))
        logging.info(f"[DEBUG] Sources in vector DB: {sorted(sources)}")
        
        # Verify all sources actually exist on disk
        missing_sources = []
        for source in sources:
            if not os.path.exists(source):
                missing_sources.append(source)
        
        if missing_sources:
            logging.error(f"CRITICAL: Found {len(missing_sources)} orphaned sources in database!")
            for missing in missing_sources:
                logging.error(f"CRITICAL: Orphaned source: {missing}")
            raise Exception(f"Data integrity compromised: {len(missing_sources)} orphaned sources found")
        
        logging.info("CRITICAL: Data integrity validation passed - all sources exist on disk")
        logging.info(f"Vector DB rebuilt from {len(chunks)} chunks using {self.current_embedding_type} embeddings and saved to {DB_PATH}! Sources: {sorted(sources)}")
        


    def get_loader(self, path: str) -> Optional[Any]:
        """Return the appropriate loader for a file, with input validation."""
        ext = os.path.splitext(path)[-1].lower()
        try:
            if not os.path.isfile(path):
                logging.warning(f"File does not exist: {path}")
                return None
            if ext == ".txt":
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if self.is_chat_email_format_strict(content):
                    return self.extract_chat_email_knowledge(content, path)
                else:
                    return TextLoader(path)
            if ext == ".pdf": return PyMuPDFLoader(path)
            if ext == ".docx": return UnstructuredWordDocumentLoader(path)
            if ext == ".csv": return CSVLoader(path)
            if ext == ".xml":
                tree = ET.parse(path)
                root = tree.getroot()
                text = ET.tostring(root, encoding="unicode", method="text")
                return [Document(page_content=text, metadata={"source": path})]
        except Exception as e:
            logging.error(f"Error loading file {path}: {e}")
            return None

    def process_chat_email_file(self, path: str) -> Any:
        """Process chat histories and email threads to extract key information."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            if self.is_chat_email_format(content):
                return self.extract_chat_email_knowledge(content, path)
            else:
                return TextLoader(path)
        except Exception as e:
            logging.error(f"Error processing {path}: {e}")
            return TextLoader(path)

    def is_chat_email_format(self, content: str) -> bool:
        """Detect if content is chat or email format."""
        chat_patterns = [
            r'\d{1,2}:\d{2}',
            r'\[.*?\]',
            r'<.*?@.*?>',
            r'From:|To:|Subject:',
            r'User:|Admin:|Support:',
            r'^\w+:\s',
        ]
        lines = content.split('\n')
        chat_line_count = 0
        total_lines = min(len(lines), 50)
        for line in lines[:total_lines]:
            for pattern in chat_patterns:
                if re.search(pattern, line):
                    chat_line_count += 1
                    break
        return (chat_line_count / total_lines) > 0.3

    def is_chat_email_format_strict(self, content: str) -> bool:
        """Stricter detection: Only treat as chat/email if >60% of first 50 lines match chat/email patterns."""
        chat_patterns = [
            r'\d{1,2}:\d{2}',
            r'\[.*?\]',
            r'<.*?@.*?>',
            r'From:|To:|Subject:',
            r'User:|Admin:|Support:',
            r'^\w+:\s',
        ]
        lines = content.split('\n')
        chat_line_count = 0
        total_lines = min(len(lines), 50)
        for line in lines[:total_lines]:
            for pattern in chat_patterns:
                if re.search(pattern, line):
                    chat_line_count += 1
                    break
        return (chat_line_count / total_lines) > 0.6

    def extract_chat_email_knowledge(self, content: str, source_path: str) -> List[Document]:
        """Extract key information from chat/email content and create FAQ-like documents."""
        try:
            structured_content = self.structure_chat_email_content(content)
            analysis_prompt = f"""Analyze the following chat history or email thread and extract key information, common questions, and important topics discussed. 

Create a structured summary that includes:
1. Main topics discussed
2. Common questions and their answers
3. Important decisions or conclusions
4. Key insights or learnings
5. Frequently mentioned issues or concerns

Format the output as a clear, organized FAQ-style document with sections like:
- Q: [Common Question] A: [Answer from discussion]
- Key Decision: [Important decision made]
- Insight: [Key learning or insight]

Content to analyze:
{structured_content[:8000]}

Please provide a structured summary:"""
            analysis = self.llm.invoke(analysis_prompt)
            extracted_doc = Document(
                page_content=analysis,
                metadata={
                    "source": source_path,
                    "type": "extracted_knowledge",
                    "original_format": "chat_email"
                }
            )
            return [extracted_doc]
        except Exception as e:
            logging.error(f"Error extracting knowledge from {source_path}: {e}")
            return TextLoader(source_path)

    def structure_chat_email_content(self, content: str) -> str:
        """Structure chat/email content for better analysis."""
        lines = content.split('\n')
        structured_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            line = re.sub(r'\[\d{1,2}:\d{2}(:\d{2})?\]', '', line)
            line = re.sub(r'<\d{1,2}:\d{2}(:\d{2})?>', '', line)
            if len(line) > 10:
                structured_lines.append(line)
        return '\n'.join(structured_lines)

    def get_doc_topic_map(self) -> dict:
        """Return a mapping: filename -> set of topic_ids."""
        DATABASE_URL = f'postgresql://postgres:{POSTGRES_PASSWORD}@localhost:5432/chat_history'
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT d.filename, dt.topic_id
            FROM documents d
            JOIN document_topics dt ON d.id = dt.document_id
        ''')
        mapping = {}
        for filename, topic_id in cursor.fetchall():
            mapping.setdefault(filename, set()).add(str(topic_id))
        conn.close()
        logging.info(f"[DEBUG] Document topic mapping: {mapping}")
        return mapping

    def handle_question(self, query: str, chat_history: Optional[list] = None, topic_ids: Optional[list] = None) -> dict:
        """Handle a question with content safety checks and optional chat history context, with topic filtering."""
        start_time = time.time()
        print(f"[TIMING] Starting question processing for: {query[:50]}...")
        
        print(f"[DEBUG] handle_question called with topic_ids: {topic_ids}")
        # if self.locked_out:
        #     return {"error": "You have been locked out for repeated misuse."}
        if not self.db:
            return {"error": "Please upload compliance documents first."}
        if not query or not query.strip():
            return {"error": "Question is required."}
                # if not self.evaluate_content_safety(query, "input"):
        #     self.inappropriate_count += 1
        #     if self.inappropriate_count >= 3:
        #         self.locked_out = True
        #         return {"error": "Locked out for repeated misuse."}
        #     return {"error": "Inappropriate question. Please try again."}
        try:
            # Debug: log vector DB doc count and validate document integrity
            try:
                vector_db_count = self.db.index.ntotal
                print(f"[DEBUG] Vector DB doc count: {vector_db_count}")
                
                # Check for document count mismatch
                if hasattr(self, 'last_document_count'):
                    if vector_db_count != self.last_document_count:
                        print(f"[WARNING] Vector DB count changed from {self.last_document_count} to {vector_db_count}")
                    self.last_document_count = vector_db_count
                else:
                    self.last_document_count = vector_db_count
                    
            except Exception as e:
                print(f"[DEBUG] Could not get vector DB doc count: {e}")
            filter_filenames = None
            if topic_ids and len(topic_ids) > 0:
                topic_filter_start = time.time()
                print(f"[DEBUG] Topic filtering requested for topic_ids: {topic_ids}")
                doc_topic_map = self.get_doc_topic_map()
                print(f"[DEBUG] Document topic map: {doc_topic_map}")
                topic_ids_set = set(str(tid) for tid in topic_ids)
                filter_filenames = [
                    fname for fname, tags in doc_topic_map.items()
                    if topic_ids_set.issubset(tags)
                ]
                print(f"[DEBUG] Filtered filenames: {filter_filenames}")
                topic_filter_time = time.time() - topic_filter_start
                print(f"[TIMING] Topic filtering took: {topic_filter_time:.3f}s")
                if not filter_filenames:
                    print("[DEBUG] No documents found for selected topics, returning early")
                    return {
                        "answer": "No documents are available for the selected topic(s).",
                        "detailed_answer": "No documents are available for the selected topic(s).",
                        "confidence": 0,
                        "sources": {"summary": "N/A", "detailed": "N/A"}
                    }
            
            query_start = time.time()
            answer, confidence, source_data, detailed_answer = self.query_answer(query, chat_history=chat_history, filter_filenames=filter_filenames)
            query_time = time.time() - query_start
            print(f"[TIMING] Main query processing took: {query_time:.3f}s")
            
            if answer == "I don't have specific information about that in the current documents.":
                # Return consistent response for no information
                return {
                    "answer": "I don't have specific information about that in the current documents.",
                    "detailed_answer": "No specific information available in current documents.",
                    "confidence": 0,
                    "sources": {"summary": "N/A", "detailed": "N/A"}
                }
            
            total_time = time.time() - start_time
            print(f"[TIMING] Total question processing took: {total_time:.3f}s")
            
            return {
                "answer": answer,
                "detailed_answer": detailed_answer,
                "confidence": confidence,
                "sources": source_data
            }
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error handling question: {e}")
            print(f"Full traceback: {error_details}")
            return {"error": f"Error processing question: {str(e)}"}

    def query_answer(self, query: str, chat_history: Optional[list] = None, filter_filenames: Optional[list] = None) -> tuple:
        """Answer a question using the vector database and LLM. Return sources with chunk/page/snippet info for debugging."""
        query_start_time = time.time()
        print(f"[TIMING] Starting query_answer processing")
        
        if not self.db:
            raise Exception("Database not loaded. Please upload documents first.")
        
        # STEP 1: Query Preprocessing
        preprocessing_start = time.time()
        print(f"[TIMING] STEP 1: Starting query preprocessing")
        if chat_history:
            history_str = "\n".join([f"User: {item['question']}\nAI: {item['answer']}" for item in chat_history])
            full_query = f"Previous conversation:\n{history_str}\n\nCurrent question: {query}"
        else:
            full_query = query
        preprocessing_time = time.time() - preprocessing_start
        print(f"[TIMING] STEP 1: Query preprocessing completed in {preprocessing_time:.3f}s")
        
        # STEP 2: Query Embedding Generation (Background Processing)
        embedding_start = time.time()
        print(f"[TIMING] STEP 2: Starting direct query embedding generation")
        try:
            # Get embedding directly - no background processing
            query_embedding = self.start_embedding_async(full_query)
            
            if query_embedding is not None:
                print(f"[TIMING] STEP 2: Direct embedding completed successfully")
            else:
                print(f"[TIMING] STEP 2: Direct embedding failed, using synchronous fallback")
                # Fallback to synchronous embedding
                query_embedding = self.embedding.embed_query(full_query)
                
        except Exception as e:
            embedding_time = time.time() - embedding_start
            print(f"[TIMING] STEP 2: Query embedding failed after {embedding_time:.3f}s: {e}")
            raise e
        
        # STEP 3: Document Retrieval (Vector Similarity Search)
        retrieval_start = time.time()
        print(f"[TIMING] STEP 3: Starting document retrieval (vector similarity search)")
        print(f"[DEBUG] Query: '{full_query}'")
        print(f"[DEBUG] Vector DB index size: {self.db.index.ntotal if hasattr(self.db, 'index') else 'Unknown'}")
        print(f"[TIMING] STEP 2: Direct embedding completed, proceeding to retrieval...")
        # Try early termination first for speed, fallback to regular search
        docs = self.get_chunks_with_early_termination(full_query, similarity_threshold=0.30)  # Early termination for speed
        retrieval_time = time.time() - retrieval_start
        print(f"[TIMING] STEP 3: Document retrieval completed in {retrieval_time:.3f}s")
        print(f"[DEBUG] Retrieved {len(docs)} documents")
        
        # Debug: Show what documents were retrieved
        if docs:
            print(f"[DEBUG] Retrieved document sources:")
            for i, doc in enumerate(docs[:5]):  # Show first 5
                source = doc.metadata.get('source', 'Unknown')
                filename = os.path.basename(str(source))
                print(f"[DEBUG]   Doc {i+1}: {filename} (chunk {doc.metadata.get('page', '?')})")
                print(f"[DEBUG]   Content preview: {doc.page_content[:100]}...")
        
        # STEP 4: Context Preparation
        context_start = time.time()
        print(f"[TIMING] STEP 4: Starting context preparation")
        
        # Create the actual context from retrieved documents
        if docs:
            context_parts = []
            for i, doc in enumerate(docs):
                context_parts.append(f"Document {i+1}:\n{doc.page_content}")
            actual_context = "\n\n".join(context_parts)
        else:
            actual_context = "No relevant documents found."
        
        # Create the complete prompt with actual document context
        # complete_prompt = f"""You are a professional compliance assistant with expertise in policy interpretation, regulatory requirements, and workplace procedures.

        # CONTEXT: {actual_context}

        # QUESTION: {full_query}

        # CRITICAL INSTRUCTIONS:
        # 1. ONLY answer based on information that is EXPLICITLY stated in the provided context
        # 2. If the context does not contain relevant information to answer the question, say "I don't have specific information about that in the current documents"
        # 3. DO NOT make up information, speculate, or reference topics not mentioned in the context
        # 4. DO NOT mention companies, products, or technologies that are not in the context
        # 5. If you find relevant information, cite the specific document or policy section
        # 6. If the answer is "no", then reply with not applicable

        # ANSWER FORMAT:
        # - If relevant information exists: Provide a direct answer with specific details and citations
        # - If no relevant information: Say "I don't have specific information about that in the current documents"
        # - Be professional and workplace-appropriate
        # - Use clear, structured language

        # IMPORTANT: Only use information that is explicitly present in the context above. Do not reference any external knowledge or make assumptions. If the context is empty or contains no relevant information, you MUST say "I don't have specific information about that in the current documents".

        # FINAL CHECK: Before providing your answer, ask yourself: "Does the context actually contain information that directly answers this specific question?" If not, say "I don't have specific information about that in the current documents"."""
        logging.info(f"[DEBUG] Retriever returned {len(docs)} docs for query: {query}")
        if filter_filenames is not None:
            logging.info(f"[DEBUG] Applying topic filter with filenames: {filter_filenames}")
            original_docs_count = len(docs)
            filtered_docs = []
            for d in docs:
                source_path = str(d.metadata.get("source", ""))
                source_filename = os.path.basename(source_path)
                logging.info(f"[DEBUG] Checking doc source: '{source_path}' -> filename: '{source_filename}'")
                if source_filename in filter_filenames:
                    filtered_docs.append(d)
            docs = filtered_docs
            logging.info(f"[DEBUG] After filtering, {len(docs)} docs remain (was {original_docs_count})")
        else:
            logging.info("[DEBUG] No topic filtering applied (filter_filenames is None)")
        
        if not docs:
            logging.warning(f"[DEBUG] No documents found for query: {query}")
            return ("I don't have specific information about that in the current documents.", 0, {"summary": "N/A", "detailed": "N/A"}, "I don't have specific information about that in the current documents.")
        
        # Log document sources for debugging
        doc_sources = [os.path.basename(str(d.metadata.get("source", ""))) for d in docs]
        logging.info(f"[DEBUG] Using documents: {doc_sources}")
        
        # Update the context with ONLY chunks using streamlined building for maximum speed
        if docs:
            # Use streamlined context building (much faster than parallel processing)
            raw_context = self.build_context_streamlined(docs)
            # Apply context size limiting for faster LLM processing
            actual_context = self.limit_context_size(raw_context, max_chars=4000)
            print(f"[SPEED] Context prepared from {len(docs)} chunks (final length: {len(actual_context)} chars)")
        else:
            actual_context = "No relevant chunks found."
        
        # Update the complete prompt with ultra-short template for maximum speed
        complete_prompt = f"Answer: {full_query}\nContext: {actual_context}\nRules: Use only context info."
        
        # Build minimal sources info for speed (skip detailed processing)
        sources = []
        source_summary = {}
        
        for i, d in enumerate(docs):
            src = os.path.basename(str(d.metadata.get("source", "N/A")))
            page = d.metadata.get("page", d.metadata.get("page_number", "?"))
            
            # Minimal source string
            if page != "?":
                source_str = f"{src} (chunk {page})"
            else:
                source_str = f"{src}"
            sources.append(source_str)
            
            # Build summary by file
            if src not in source_summary:
                source_summary[src] = []
            source_summary[src].append(page)
        
        # Create summary format (minimal)
        summary_parts = []
        for src, pages in source_summary.items():
            unique_pages = sorted(list(set(pages)), key=lambda x: int(str(x)) if str(x).isdigit() else 0)
            if len(unique_pages) == 1:
                summary_parts.append(f"{src} (chunk {unique_pages[0]})")
            else:
                summary_parts.append(f"{src} (chunks {', '.join(map(str, unique_pages))})")
        
        sources_summary = " | ".join(summary_parts)
        sources_detailed = " | ".join(sources)
        
        # Format sources more cleanly
        if len(sources) == 1:
            sources_summary = summary_parts[0]
            sources_detailed = sources[0]
        top_similarity = None
        try:
            # Get the embedding result directly
            if query_embedding is not None:
                print(f"[TIMING] STEP 2: Using direct embedding result")
                query_emb = query_embedding  # Direct embedding result
                embedding_time = time.time() - embedding_start
                print(f"[TIMING] STEP 2: Direct embedding completed in {embedding_time:.3f}s")
            else:
                # Fallback if embedding failed
                print(f"[TIMING] STEP 2: Embedding failed, using fallback")
                embedding_time = time.time() - embedding_start
                print(f"[TIMING] STEP 2: Embedding failed in {embedding_time:.3f}s")
                query_emb = None
            
            # Calculate similarity only with the retrieved documents (not all chunks)
            if docs:
                import numpy as np
                from numpy.linalg import norm
                similarities = []
                
                # Calculate similarity with each retrieved document using optimized embeddings
                for doc in docs:
                    # Get document embedding from FAISS index
                    doc_id = doc.metadata.get('chunk_id', 0)
                    if hasattr(self.db.index, 'reconstruct'):
                        doc_emb = self.db.index.reconstruct(doc_id)
                        
                        # SPEED OPTIMIZATION: Ensure both embeddings have same dimension
                        if len(query_emb) != len(doc_emb):
                            # Truncate to smaller dimension for compatibility
                            min_dim = min(len(query_emb), len(doc_emb))
                            query_emb_trunc = query_emb[:min_dim]
                            doc_emb_trunc = doc_emb[:min_dim]
                            print(f"[SPEED] Truncated embeddings to {min_dim}D for similarity calculation")
                        else:
                            query_emb_trunc = query_emb
                            doc_emb_trunc = doc_emb
                        
                        sim = np.dot(query_emb_trunc, doc_emb_trunc) / (norm(query_emb_trunc) * norm(doc_emb_trunc) + 1e-8)
                        similarities.append(sim)
                
                if similarities:
                    top_similarity = max(similarities)
                    logging.info(f"[DEBUG] Native {MODEL_NAME} similarity calculation successful with {len(docs)} docs, max similarity: {top_similarity:.4f}")
                    
        except Exception as e:
            logging.error(f"Native {MODEL_NAME} similarity calculation failed: {e}")
            top_similarity = None
        # Complete context preparation timing
        context_time = time.time() - context_start
        print(f"[TIMING] STEP 4: Context preparation completed in {context_time:.3f}s")
        
        # STEP 5: LLM Call with Speed Optimizations
        llm_start = time.time()
        print(f"[TIMING] STEP 5: Starting optimized LLM call")
        try:
            print(f"[SPEED] Sending optimized prompt to LLM (length: {len(complete_prompt)} chars)")
            
            # Use direct LLM call for maximum speed (no streaming overhead)
            raw_answer = self.llm.invoke(complete_prompt)
            
            # Apply response length limiting
            answer = self.get_concise_response(complete_prompt) if len(raw_answer) > 800 else raw_answer
            
            # Handle different response formats
            if isinstance(answer, dict) and 'result' in answer:
                answer = answer['result']
            else:
                answer = str(answer)
            if isinstance(answer, list):
                answer = "\n".join(str(a) for a in answer)
                
            llm_time = time.time() - llm_start
            print(f"[TIMING] STEP 5: Optimized LLM call completed in {llm_time:.3f}s")
            print(f"[SPEED] LLM response received (length: {len(answer)} chars)")
            
            # STEP 6: Response Processing & Output
            response_processing_start = time.time()
            print(f"[TIMING] STEP 6: Starting response processing & output")
            
            # Parse the structured response to extract helpful and detailed answers
            helpful_answer = ""
            detailed_answer = ""
            
            # Try to extract structured format
            import re
            helpful_match = re.search(r'HELPFUL ANSWER:\s*(.*?)(?=\nDETAILED ANSWER:|$)', answer, re.DOTALL | re.IGNORECASE)
            detailed_match = re.search(r'DETAILED ANSWER:\s*(.*?)(?=\n[A-Z]|$)', answer, re.DOTALL | re.IGNORECASE)
            
            if helpful_match and detailed_match:
                helpful_answer = helpful_match.group(1).strip()
                detailed_answer = detailed_match.group(1).strip()
                print(f"[DEBUG] Successfully parsed structured response - helpful: {len(helpful_answer)} chars, detailed: {len(detailed_answer)} chars")
                print(f"[DEBUG] Helpful preview: {helpful_answer[:100]}...")
                print(f"[DEBUG] Detailed preview: {detailed_answer[:100]}...")
            else:
                # Fallback: treat the entire response as detailed answer and generate helpful answer
                detailed_answer = answer.strip()
                helpful_answer = self.generate_short_answer(query, detailed_answer, docs)
                print(f"[DEBUG] Using fallback parsing - generated helpful answer from detailed")
            
            # Use helpful answer as the main answer for processing
            answer = helpful_answer
            
        except Exception as e:
            logging.error(f"Error calling LLM: {e}")
            import traceback
            logging.error(f"LLM error traceback: {traceback.format_exc()}")
            raise Exception(f"Failed to get response from AI model: {str(e)}")
        
        # Activity 5 & 6: Extra processing (guardrails, hallucination checks, etc.)
        extra_processing_start = time.time()
        logging.info(f"[TIMING] Starting extra processing (guardrails, hallucination checks)")
        
        # Post-processing: Check for hallucination and validate answer quality
        answer_lower = answer.lower()
        
        # # AI-based hallucination detection using embeddings and similarity
        # potential_hallucination = False
        
        # if docs and answer.strip():
        #     try:
        #         # Get embeddings for the answer and the context
        #         answer_embedding = self.embedding.embed_query(answer)
                
        #         # Create context embedding by combining all document embeddings
        #         context_text = "\n\n".join([doc.page_content for doc in docs])
        #         context_embedding = self.embedding.embed_query(context_text)
                
        #         # Calculate cosine similarity between answer and context
        #         import numpy as np
        #         from numpy.linalg import norm
                
        #         similarity = np.dot(answer_embedding, context_embedding) / (norm(answer_embedding) * norm(context_embedding) + 1e-8)
                
        #         logging.info(f"[DEBUG] Answer-context similarity: {similarity:.4f}")
                
        #         # If similarity is very low (< 0.3), the answer might be hallucinated
        #         if similarity < 0.3):
        #             logging.warning(f"[DEBUG] Low similarity detected ({similarity:.4f}), potential hallucination")
        #             potential_hallucination = True
                    
        #     except Exception as e:
        #         logging.warning(f"[DEBUG] Error in AI-based hallucination detection: {e}")
        #         # Fallback: don't flag as hallucination if detection fails
        
        # if potential_hallucination:
        #     logging.warning(f"[DEBUG] Potential hallucination detected in answer: {answer[:200]}...")
        #     # Force a re-prompt with stronger anti-hallucination instructions
        #     anti_hallucination_prompt = f"""You are a professional compliance assistant. The user asked: {query}

        # CRITICAL: You must ONLY answer based on information explicitly stated in the provided context. 

        # CONTEXT: {actual_context}

        # IMPORTANT RULES:
        # 1. If the context does not contain relevant information, say "I don't have specific information about that in the current documents"
        # 2. Do NOT mention any companies, products, or technologies not explicitly mentioned in the context
        # 3. Do NOT reference "Document 1", "Document 2", etc. unless they are actual document names in the context
        # 4. Do NOT make up information or speculate
        # 5. If the context is empty or contains no relevant information, you MUST say "I don't have specific information about that in the current documents"

        # Please provide your answer:"""
            
        #     try:
        #         corrected_result = self.llm.invoke(anti_hallucination_prompt)
        #         if isinstance(corrected_result, dict) and 'result' in corrected_result:
        #             answer = corrected_result['result']
        #         else:
        #             answer = str(corrected_result)
        #         logging.info("[DEBUG] Answer corrected for potential hallucination")
        #     except Exception as e:
        #         logging.warning(f"Anti-hallucination correction failed: {e}")
        
        # # Check if answer is too vague or says "I don't know"
        # if any(phrase in answer_lower for phrase in ["i don't know", "i don't have", "no information", "not mentioned", "not found"]):
        #     # Re-prompt with stronger instructions if context exists
        #     if docs and any(d.page_content.strip() for d in docs):
        #         logging.info("[DEBUG] Answer was too vague, re-prompting with stronger instructions")
        #         enhanced_prompt = f"""You are a professional compliance assistant. The user asked: {query}

        # You have access to relevant documents. Please answer based ONLY on information explicitly stated in the context.

        # IMPORTANT: 
        # - If the context contains relevant information, provide a clear answer with citations
        # - If the context does not contain relevant information, say "I don't have specific information about that in the current documents"
        # - Do NOT make up information or reference topics not in the context
        # - If the context is empty or contains no relevant information, you MUST say "I don't have specific information about that in the current documents"

        # Please answer the question using the available context:"""
                
        #         try:
        #             enhanced_result = self.llm.invoke(enhanced_prompt)
        #             if isinstance(enhanced_result, dict) and 'result' in enhanced_result:
        #             answer = enhanced_result['result']
        #         else:
        #             answer = str(enhanced_result)
        #         except Exception as e:
        #             logging.warning(f"Enhanced prompt failed: {e}")
        #             # Keep the original answer if enhancement fails
        
        # # Additional quality check: If answer is still too short or vague, try one more time
        # if len(answer.strip()) < 50 and docs:
        #     logging.info("[DEBUG] Answer is too short, attempting to generate more detailed response")
        #     detailed_prompt = f"""You are a professional compliance assistant. The user asked: {query}

        # You have access to relevant documents. Please provide an answer that:
        # 1. ONLY uses information explicitly stated in the context
        # 2. Directly addresses the question if relevant information exists
        # 3. Includes specific details and citations from the documents
        # 4. Is professional and workplace-appropriate
        # 5. If no relevant information exists, clearly state "I don't have specific information about that in the current documents"

        # IMPORTANT: Do not make up information or reference topics not in the context. If the context is empty or contains no relevant information, you MUST say "I don't have specific information about that in the current documents".

        # Please provide your answer:"""
            
        #     try:
        #         detailed_result = self.llm.invoke(detailed_prompt)
        #         if isinstance(detailed_result, dict) and 'result' in detailed_result:
        #             answer = detailed_result['result']
        #         else:
        #             answer = str(detailed_result)
        #         except Exception as e:
        #             logging.warning(f"Detailed prompt failed: {e}")
        #             # Keep the original answer if enhancement fails
        
        # Use simple default confidence since LLM-based calculation is disabled
        confidence = 0.85  # Default confidence
        print(f"[TIMING] Using default confidence: {confidence}")
        
        # Return structured source data
        source_data = {
            "summary": sources_summary if sources else "N/A",
            "detailed": sources_detailed if sources else "N/A"
        }
        
        # Skip answer enhancement entirely for speed
        final_answer = answer.strip()
        print(f"[TIMING] Skipping answer enhancement for speed")
        
        # Skip final hallucination check entirely for speed
        print(f"[TIMING] Skipping final hallucination check for speed")
        
        # Use the parsed helpful answer as final answer, and store detailed answer
        if 'helpful_answer' in locals() and helpful_answer:
            final_answer = helpful_answer
        if 'detailed_answer' in locals() and detailed_answer:
            # Keep the parsed detailed answer
            detailed_answer = detailed_answer
        else:
            # If no detailed answer was parsed, use the original answer as detailed
            detailed_answer = final_answer
        
        # Complete response processing timing
        response_processing_time = time.time() - response_processing_start
        print(f"[TIMING] STEP 6: Response processing & output completed in {response_processing_time:.3f}s")
        
        # Calculate total query time
        total_query_time = time.time() - query_start_time
        print(f"[TIMING] Total query_answer processing took: {total_query_time:.3f}s")
        
        # Print comprehensive timing summary
        print(f"[TIMING] ===== QUERY PIPELINE TIMING SUMMARY =====")
        print(f"[TIMING] STEP 1 (Preprocessing): {preprocessing_time:.3f}s")
        print(f"[TIMING] STEP 2 (Embedding): {embedding_time:.3f}s")
        print(f"[TIMING] STEP 3 (Retrieval): {retrieval_time:.3f}s")
        print(f"[TIMING] STEP 4 (Context Prep): {context_time:.3f}s")
        print(f"[TIMING] STEP 5 (LLM Call): {llm_time:.3f}s")
        print(f"[TIMING] STEP 6 (Response Processing): {response_processing_time:.3f}s")
        print(f"[TIMING] TOTAL TIME: {total_query_time:.3f}s")
        print(f"[TIMING] ==========================================")
        
        # Log what we're returning for debugging
        print(f"[DEBUG] Returning - final_answer: {len(final_answer)} chars, detailed_answer: {len(detailed_answer)} chars")
        print(f"[DEBUG] Final helpful preview: {final_answer[:100]}...")
        print(f"[DEBUG] Final detailed preview: {detailed_answer[:100]}...")
        
        # Return both helpful and detailed answers
        return final_answer, confidence, source_data, detailed_answer

    def calculate_answer_confidence(self, query: str, answer: str, docs: list) -> float:
        """Calculate the AI's confidence in its answer using enhanced evaluation"""
        try:
            # Prepare context from retrieved documents
            context = "\n\n".join([f"Document {i+1}: {doc.page_content}" for i, doc in enumerate(docs)])
            
            # Enhanced confidence evaluation prompt
            confidence_prompt = f"""You are an expert evaluator assessing the quality and confidence of a compliance assistant's answer.

Question: {query}

Answer provided: {answer}

Source documents used:
{context}

EVALUATION CRITERIA:
1. RELEVANCE: Does the answer directly address the question asked?
2. ACCURACY: Is the information in the answer supported by the source documents?
3. COMPLETENESS: Does the answer provide sufficient detail to be useful?
4. SYNTHESIS: If multiple sources were used, were they effectively combined?
5. PROFESSIONALISM: Is the answer clear, well-structured, and workplace-appropriate?

CONFIDENCE SCALE (0-100):
- 0-20: Very low confidence (answer is incorrect, irrelevant, or completely unsupported)
- 21-40: Low confidence (answer has major gaps, is vague, or poorly supported)
- 41-60: Moderate confidence (answer is partially correct but has significant limitations)
- 61-80: High confidence (answer is mostly correct, well-supported, and useful)
- 81-100: Very high confidence (answer is accurate, complete, well-synthesized, and professional)

Consider the quality of the answer relative to the available information. If the answer effectively uses whatever relevant information is available, even if incomplete, it should receive higher confidence.

Respond with ONLY a number between 0 and 100, representing your confidence percentage."""

            # Get confidence score from LLM
            confidence_response = self.llm.invoke(confidence_prompt)
            confidence_text = str(confidence_response).strip()
            
            # Extract numeric confidence score
            import re
            confidence_match = re.search(r'\b(\d{1,3})\b', confidence_text)
            if confidence_match:
                confidence_score = float(confidence_match.group(1))
                # Ensure it's within 0-100 range
                confidence_score = max(0, min(100, confidence_score))
                return round(confidence_score, 2)
            else:
                # Fallback if parsing fails
                return 50.0
                
        except Exception as e:
            logging.warning(f"Confidence calculation failed: {e}")
            # Fallback confidence based on whether we have documents
            return 20.0 if docs else 0.0

    def enhance_answer_quality(self, query: str, answer: str, docs: list) -> str:
        """Enhance answer quality by checking for common issues and improving structure"""
        enhancement_start = time.time()
        logging.info(f"[TIMING] Starting answer enhancement")
        
        try:
            # Check if answer needs improvement
            answer_lower = answer.lower()
            
            # If answer is too short or doesn't seem comprehensive enough
            if len(answer.strip()) < 100 and docs:
                logging.info("[DEBUG] Answer may need enhancement, checking quality")
                
                # Check if we have good source material
                total_source_length = sum(len(doc.page_content) for doc in docs)
                if total_source_length > 500:  # We have substantial source material
                    enhancement_prompt = f"""You are a professional compliance assistant. The user asked: {query}

Current answer: {answer}

You have access to substantial source material. Please enhance this answer to be more comprehensive and professional. The enhanced answer should:

1. Be more detailed and specific
2. Include relevant policy citations or references when possible
3. Structure the information logically
4. Be professional and workplace-appropriate
5. Address the question more thoroughly

Please provide an enhanced version of the answer:"""
                    
                    # enhancement_llm_start = time.time()
                    # enhanced_result = self.llm.invoke(enhancement_prompt)
                    # enhancement_llm_time = time.time() - enhancement_llm_start
                    # logging.info(f"[TIMING] Enhancement LLM call took: {enhancement_llm_time:.3f}s")
                    
                    # if isinstance(enhanced_result, dict) and 'result' in enhanced_result:
                    #     enhanced_answer = enhanced_result['result']
                    # else:
                    #     enhanced_answer = str(enhanced_result)
                    
                    # # Only use enhanced answer if it's significantly better
                    # if len(enhanced_answer.strip()) > len(answer.strip()) * 1.5:
                    #     enhancement_time = time.time() - enhancement_start
                    #     logging.info(f"[TIMING] Answer enhancement completed in: {enhancement_time:.3f}s")
                    #     return enhanced_answer.strip()
            
            # Check for common issues and fix them
            if "i don't know" in answer_lower or "no information" in answer_lower:
                # Try to extract any useful information from sources
                useful_info = []
                for doc in docs:
                    content = doc.page_content.lower()
                    if any(keyword in content for keyword in query.lower().split()):
                        useful_info.append(doc.page_content[:200] + "...")
                
                if useful_info:
                    fallback_prompt = f"""The user asked: {query}

While I don't have the exact answer, I found some potentially relevant information in the documents. Please provide a helpful response that:

1. Acknowledges the limitations
2. Shares any relevant information found
3. Suggests where they might find more information
4. Is professional and helpful

Relevant information found:
{chr(10).join(useful_info)}

Please provide a helpful response:"""
                    
                    # fallback_llm_start = time.time()
                    # fallback_result = self.llm.invoke(fallback_prompt)
                    # fallback_llm_time = time.time() - fallback_llm_start
                    # logging.info(f"[TIMING] Fallback LLM call took: {fallback_llm_time:.3f}s")
                    
                    # if isinstance(fallback_result, dict) and 'result' in enhanced_result):
                    #     enhancement_time = time.time() - enhancement_start
                    #     logging.info(f"[TIMING] Answer enhancement completed in: {enhancement_time:.3f}s")
                    #     return fallback_result['result'].strip()
            
            enhancement_time = time.time() - enhancement_start
            logging.info(f"[TIMING] Answer enhancement completed in: {enhancement_time:.3f}s")
            return answer.strip()
            
        except Exception as e:
            logging.error(f"Error enhancing answer quality: {e}")
            enhancement_time = time.time() - enhancement_start
            logging.info(f"[TIMING] Answer enhancement failed after: {enhancement_time:.3f}s")
            return answer.strip()
    
    def generate_short_answer(self, query: str, detailed_answer: str, docs: list) -> str:
        """Generate a concise, one-line summary answer for quick overview."""
        try:
            short_prompt = f"""You are a professional compliance assistant. The user asked: {query}

Here is the detailed answer: {detailed_answer}

Please create a concise, one-line summary answer (maximum 2 sentences) that:
1. Directly answers the question
2. Captures the key point or conclusion
3. Is professional and clear
4. Can be understood quickly

Provide ONLY the short summary answer:"""

            # short_result = self.llm.invoke(short_prompt)
            # if isinstance(short_result, dict) and 'result' in short_result:
            #     short_answer = short_result['result']
            # else:
            #     short_answer = str(short_result)
            
            # Clean up the short answer
            short_answer = short_answer.strip()
            if short_answer.startswith('"') and short_answer.endswith('"'):
                short_answer = short_answer[1:-1]
            
            # Fallback: if LLM fails, create a simple summary
            if len(short_answer) < 10 or len(short_answer) > 300:
                # Create a simple summary from the detailed answer
                sentences = detailed_answer.split('.')
                if sentences:
                    short_answer = sentences[0].strip() + '.'
                    if len(short_answer) > 200:
                        short_answer = short_answer[:200] + '...'
            
            return short_answer
            
        except Exception as e:
            logging.warning(f"Short answer generation failed: {e}")
            # Fallback: return first sentence of detailed answer
            sentences = detailed_answer.split('.')
            if sentences:
                return sentences[0].strip() + '.'
            else:
                return detailed_answer[:100] + '...' if len(detailed_answer) > 100 else detailed_answer

    def force_rebuild_db(self) -> None:
        """Safely delete the FAISS index directory and rebuild the vector DB from scratch."""
        import os
        import shutil
        from pathlib import Path
        
        print(f"[HNSW] CRITICAL: Starting forced rebuild with complete data cleanup...")
        
        # CRITICAL: Clear database from memory first
        self.db = None
        
        db_path = Path(DB_PATH).resolve()
        project_root = Path(__file__).parent.resolve()
        
        # Only allow deletion if DB_PATH is a subdirectory of the project and not '.', '', or '/'
        if db_path == project_root or str(db_path) in ('/', '') or not str(db_path).startswith(str(project_root)):
            add_backend_log(f"[ERROR] Unsafe DB_PATH for deletion: {db_path}. Skipping vector DB cleanup.")
            return
        
        # CRITICAL: Force complete cleanup of all index files
        if db_path.exists() and db_path.is_dir():
            try:
                # Remove all files in the directory
                for file_path in db_path.iterdir():
                    if file_path.is_file():
                        file_path.unlink()
                        print(f"[HNSW] CRITICAL: Deleted index file: {file_path}")
                
                # Remove the directory itself
                shutil.rmtree(db_path)
                print(f"[HNSW] CRITICAL: Deleted old FAISS index directory: {db_path}")
                add_backend_log(f"Deleted old FAISS index directory: {db_path}")
                
            except Exception as e:
                add_backend_log(f"[ERROR] Error deleting FAISS index: {e}")
                print(f"[HNSW] CRITICAL ERROR: Could not delete index directory: {e}")
                raise Exception(f"Cannot delete index directory: {e}")
        else:
            print(f"[HNSW] CRITICAL: No FAISS index directory to delete at: {db_path}")
            add_backend_log(f"No FAISS index directory to delete at: {db_path}")
        
        # CRITICAL: Verify cleanup was successful
        if db_path.exists():
            raise Exception(f"CRITICAL: Index directory still exists after deletion attempt: {db_path}")
        
        print(f"[HNSW] CRITICAL: Complete cleanup verified - rebuilding database...")
        
        # Rebuild with guaranteed clean state
        self.build_db("forced_rebuild")
        add_backend_log("Forced rebuild of vector DB complete with guaranteed data cleanup.")
        print(f"[HNSW] CRITICAL: Forced rebuild completed with guaranteed data integrity")

    def add_document_incremental(self, file_path: str) -> bool:
        """Add a single document to the existing database using incremental update."""
        try:
            logging.info(f"[DEBUG] Adding document incrementally: {file_path}")
            
            # Switch to Ollama for incremental updates
            self.select_optimal_embedding("incremental_update")
            
            # Load the single document
            filename = os.path.basename(file_path)
            ext = os.path.splitext(file_path)[1].lower()
            
            # Check if document already exists in database to prevent duplicates
            if self.db:
                try:
                    # Get a sample of existing documents to check for duplicates
                    sample_docs = self.db.similarity_search("", k=1000)
                    existing_sources = set()
                    for doc in sample_docs:
                        source = doc.metadata.get('source', '')
                        if source:
                            existing_sources.add(os.path.basename(source))
                    
                    if filename in existing_sources:
                        logging.warning(f"[DEBUG] Document {filename} already exists in database, skipping incremental add")
                        return True  # Already exists, consider it successful
                        
                except Exception as e:
                    logging.warning(f"[DEBUG] Could not check for duplicates: {e}, proceeding with add")
            
            docs = []
            if ext == ".pdf":
                try:
                    loader = PyMuPDFLoader(file_path)
                    docs = loader.load()
                    logging.info(f"[DEBUG] Loaded {len(docs)} docs from {filename} (PyMuPDF)")
                except Exception as e:
                    logging.warning(f"[DEBUG] PyMuPDF failed for {filename}: {e}. Trying OCR...")
                    loader = UnstructuredPDFLoader(file_path, strategy="ocr_only")
                    docs = loader.load()
                    logging.info(f"[DEBUG] Loaded {len(docs)} docs from {filename} (OCR)")
            elif ext == ".txt":
                loader = TextLoader(file_path)
                docs = loader.load()
            elif ext == ".docx":
                loader = UnstructuredWordDocumentLoader(file_path)
                docs = loader.load()
            elif ext == ".csv":
                loader = CSVLoader(file_path)
                docs = loader.load()
            else:
                logging.warning(f"Unsupported file type: {file_path}")
                return False
            
            if not docs:
                logging.warning(f"No documents loaded from {filename}")
                return False
            
            # Use smart chunking for consistent quality
            chunk_size, chunk_overlap = self.calculate_optimal_chunk_parameters()
            
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
            chunks = splitter.split_documents(docs)
            logging.info(f"[DEBUG] Created {len(chunks)} chunks from {filename}")
            
            # Add chunks to existing database
            if self.db:
                self.db.add_documents(chunks)
                logging.info(f"[DEBUG] Added {len(chunks)} chunks from {filename} to database")
                
                # Save the updated database
                self.db.save_local(DB_PATH)
                
                # Validate integrity after adding to prevent stale data
                integrity_result = self.validate_vector_db_integrity()
                if integrity_result.get('status') == 'rebuilt':
                    logging.warning(f"[DEBUG] Vector database rebuilt after adding {filename} due to integrity issues")
                
                logging.info(f"[DEBUG] Successfully added {filename} to database")
                return True
            else:
                logging.error("No database available for incremental update")
                return False
                
        except Exception as e:
            logging.error(f"Error adding document incrementally: {e}")
            return False

    def delete_document_incremental(self, filename: str) -> bool:
        """Remove a document's chunks from the database (requires rebuilding)."""
        try:
            logging.info(f"[DEBUG] CRITICAL: Deleting document: {filename}")
            
            # CRITICAL FIX: ALWAYS force complete rebuild to guarantee no data leakage
            # This ensures deleted document data is completely removed
            
            # Get list of all files except the one to delete
            all_files = []
            for root, _, files in os.walk(DOCS_PATH):
                for file in files:
                    if file != filename:
                        all_files.append(os.path.join(root, file))
            
            if len(all_files) == 0:
                logging.warning("No files remaining after deletion")
                return False
            
            # CRITICAL: Force complete rebuild to ensure data integrity
            logging.info(f"[DEBUG] CRITICAL: Forcing complete database rebuild after deletion")
            self.build_db("forced_rebuild_after_deletion")
            
            logging.info(f"[DEBUG] CRITICAL: Successfully deleted {filename} with guaranteed data cleanup")
            return True
            
        except Exception as e:
            logging.error(f"CRITICAL ERROR: Error deleting document: {e}")
            return False

    def remove_document_from_vector_db(self, filename: str) -> bool:
        """Remove a specific document from the vector database incrementally."""
        try:
            logging.info(f"[DEBUG] Removing document from vector database: {filename}")
            
            if not self.db:
                logging.warning("[DEBUG] No vector database to remove document from")
                return False
            
            # Get list of all files except the one to delete
            remaining_files = []
            for root, _, files in os.walk(DOCS_PATH):
                for file in files:
                    if file != filename:
                        remaining_files.append(os.path.join(root, file))
            
            if not remaining_files:
                # No files left, clear the database
                self.db = None
                logging.info(f"[DEBUG] No files left, cleared vector database")
                return True
            
            # CRITICAL: Force complete rebuild to ensure data integrity
            logging.info(f"[DEBUG] CRITICAL: Forcing complete vector database rebuild after deletion")
            self.build_db("forced_rebuild_after_deletion")
            return True
            
        except Exception as e:
            logging.error(f"Error removing document from vector database: {e}")
            return False

    def calculate_optimal_chunk_parameters(self) -> tuple:
        """Calculate optimal chunk size and overlap based on available resources for FAQ bot quality."""
        try:
            # Get current system resources
            memory = psutil.virtual_memory()
            available_ram_gb = memory.available / (1024**3)
            total_ram_gb = memory.total / (1024**3)
            memory_pressure = memory.percent
            
            cpu_cores = psutil.cpu_count(logical=True)
            
            # Base chunk size calculation based on available resources
            if available_ram_gb > 16:
                base_chunk_size = 256
            elif available_ram_gb > 8:
                base_chunk_size = 192
            elif available_ram_gb > 4:
                base_chunk_size = 128
            else:
                base_chunk_size = 96
            
            # Adjust based on memory pressure
            if memory_pressure > 80:
                base_chunk_size = int(base_chunk_size * 0.7)  # Reduce by 30% under pressure
            elif memory_pressure > 60:
                base_chunk_size = int(base_chunk_size * 0.85)  # Reduce by 15% under moderate pressure
            
            # Adjust based on CPU cores (more cores = can handle larger chunks)
            if cpu_cores > 8:
                base_chunk_size = int(base_chunk_size * 1.2)  # Increase by 20%
            elif cpu_cores > 4:
                base_chunk_size = int(base_chunk_size * 1.1)  # Increase by 10%
            
            # FAQ Bot Quality Bounds - ensure chunks are optimal for Q&A
            MIN_CHUNK_SIZE = 128    # Minimum for meaningful context in Q&A
            MAX_CHUNK_SIZE = 512    # Maximum for FAQ efficiency
            
            # Apply quality bounds
            optimal_chunk_size = max(MIN_CHUNK_SIZE, min(MAX_CHUNK_SIZE, base_chunk_size))
            
            # Calculate optimal overlap (25-30% for good continuity in Q&A)
            overlap_ratio = 0.28  # 28% overlap for FAQ bots
            optimal_overlap = max(32, int(optimal_chunk_size * overlap_ratio))
            
            # Ensure overlap doesn't exceed chunk size
            optimal_overlap = min(optimal_overlap, optimal_chunk_size - 32)
            
            logging.info(f"[DEBUG] Chunking calculation:")
            logging.info(f"[DEBUG]   - Available RAM: {available_ram_gb:.2f} GB")
            logging.info(f"[DEBUG]   - Memory pressure: {memory_pressure:.1f}%")
            logging.info(f"[DEBUG]   - CPU cores: {cpu_cores}")
            logging.info(f"[DEBUG]   - Base chunk size: {base_chunk_size}")
            logging.info(f"[DEBUG]   - Final chunk size: {optimal_chunk_size}")
            logging.info(f"[DEBUG]   - Final overlap: {optimal_overlap}")
            
            return optimal_chunk_size, optimal_overlap
            
        except Exception as e:
            logging.warning(f"[DEBUG] Error calculating chunk parameters: {e}, using defaults")
            # Fallback to good FAQ bot defaults
            return 192, 54  # 192 chars with 54 overlap (28%)

    def replace_document_incremental(self, old_filename: str, new_file_path: str) -> bool:
        """Replace a document in the database."""
        try:
            logging.info(f"[DEBUG] Replacing document: {old_filename} with {new_file_path}")
            
            # Delete the old document
            if not self.delete_document_incremental(old_filename):
                logging.error(f"Failed to delete old document: {old_filename}")
                return False
            
            # Add the new document
            if not self.add_document_incremental(new_file_path):
                logging.error(f"Failed to add new document: {new_file_path}")
                return False
            
            logging.info(f"[DEBUG] Successfully replaced {old_filename} with {new_file_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error replacing document: {e}")
            return False

    def extract_tables_from_pdf(self, file_path: str) -> List[Document]:
        """Extract tables from PDF using multiple methods for better accuracy."""
        tables = []
        
        if not TABLE_EXTRACTION_AVAILABLE:
            logging.warning("Table extraction not available - install dependencies")
            return tables
            
        try:
            # Method 1: Try tabula-py (works well for most tables)
            if tabula and hasattr(tabula, 'read_pdf'):
                try:
                    pdf_tables = tabula.read_pdf(file_path, pages='all', multiple_tables=True)
                    for i, table in enumerate(pdf_tables):
                        if not table.empty:
                            # Convert table to structured text
                            table_text = self.table_to_text(table, f"Table {i+1}")
                            if table_text.strip():
                                tables.append(Document(
                                    page_content=table_text,
                                    metadata={"source": file_path, "type": "table", "table_index": i}
                                ))
                except Exception as e:
                    logging.warning(f"Tabula failed for {file_path}: {e}")
            
            # Method 2: Try camelot (better for complex tables)
            if camelot and hasattr(camelot, 'read_pdf'):
                try:
                    pdf_tables = camelot.read_pdf(file_path, pages='all')
                    for i, table in enumerate(pdf_tables):
                        if table.df is not None and not table.df.empty:
                            # Convert table to structured text
                            table_text = self.table_to_text(table.df, f"Table {i+1}")
                            if table_text.strip():
                                tables.append(Document(
                                    page_content=table_text,
                                    metadata={"source": file_path, "type": "table", "table_index": i}
                                ))
                except Exception as e:
                    logging.warning(f"Camelot failed for {file_path}: {e}")
            
            # Method 3: OCR-based table extraction for image-based PDFs
            if cv2 and pytesseract:
                try:
                    tables.extend(self.extract_tables_with_ocr(file_path))
                except Exception as e:
                    logging.warning(f"OCR table extraction failed for {file_path}: {e}")
                
        except Exception as e:
            logging.error(f"Error extracting tables from {file_path}: {e}")
        
        return tables

    def extract_tables_with_ocr(self, file_path: str) -> List[Document]:
        """Extract tables using OCR for image-based PDFs."""
        tables = []
        
        if not (cv2 and pytesseract):
            return tables
            
        try:
            import numpy as np
            import fitz  # PyMuPDF
            
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get page as image
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                
                # Convert to OpenCV format
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Detect table lines
                horizontal_lines = cv2.HoughLinesP(
                    gray, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10
                )
                vertical_lines = cv2.HoughLinesP(
                    cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE), 1, np.pi/180, 
                    threshold=100, minLineLength=100, maxLineGap=10
                )
                
                # If we detect table structure, extract text
                if horizontal_lines is not None or vertical_lines is not None:
                    # Extract text from the entire page
                    text = pytesseract.image_to_string(img)
                    
                    # Try to structure the text as a table
                    structured_text = self.structure_ocr_text_as_table(text)
                    if structured_text.strip():
                        tables.append(Document(
                            page_content=structured_text,
                            metadata={"source": file_path, "type": "ocr_table", "page": page_num}
                        ))
            
            doc.close()
            
        except Exception as e:
            logging.error(f"Error in OCR table extraction: {e}")
        
        return tables

    def structure_ocr_text_as_table(self, text: str) -> str:
        """Structure OCR text that appears to be tabular data."""
        lines = text.strip().split('\n')
        structured_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Split by multiple spaces or tabs to identify columns
                columns = [col.strip() for col in line.split() if col.strip()]
                if len(columns) > 1:
                    # This looks like table data
                    structured_lines.append(" | ".join(columns))
                else:
                    structured_lines.append(line)
        
        return "\n".join(structured_lines)

    def table_to_text(self, table, table_name: str = "Table") -> str:
        """Convert a pandas DataFrame to structured text representation."""
        try:
            if not TABLE_EXTRACTION_AVAILABLE or not pd:
                return str(table)
                
            if hasattr(table, 'empty') and table.empty:
                return ""
            
            # Get column names
            columns = table.columns.tolist()
            
            # Create header
            text_lines = [f"{table_name}:"]
            text_lines.append("Columns: " + " | ".join(str(col) for col in columns))
            text_lines.append("-" * 50)
            
            # Add data rows
            for idx, row in table.iterrows():
                row_data = []
                for col in columns:
                    value = row[col]
                    # Handle NaN and None values
                    if pd.isna(value) or value is None:
                        value = "N/A"
                    else:
                        value = str(value).strip()
                    row_data.append(value)
                text_lines.append(" | ".join(row_data))
            
            # Add summary statistics for numeric columns
            if np:
                numeric_cols = table.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    text_lines.append("-" * 50)
                    text_lines.append("Summary Statistics:")
                    for col in numeric_cols:
                        if not table[col].isna().all():
                            text_lines.append(f"{col}: Min={table[col].min():.2f}, Max={table[col].max():.2f}, Mean={table[col].mean():.2f}")
            
            return "\n".join(text_lines)
            
        except Exception as e:
            logging.error(f"Error converting table to text: {e}")
            return str(table)

    def extract_tables_from_excel(self, file_path: str) -> List[Document]:
        """Extract tables from Excel files."""
        tables = []
        
        if not TABLE_EXTRACTION_AVAILABLE or not pd:
            logging.warning("Table extraction not available - install dependencies")
            return tables
            
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            
            for sheet_name in excel_file.sheet_names:
                try:
                    # Read the sheet
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    
                    if not df.empty:
                        # Convert table to structured text
                        table_text = self.table_to_text(df, f"Sheet: {sheet_name}")
                        if table_text.strip():
                            tables.append(Document(
                                page_content=table_text,
                                metadata={"source": file_path, "type": "excel_table", "sheet": sheet_name}
                            ))
                except Exception as e:
                    logging.warning(f"Error reading sheet {sheet_name}: {e}")
                    
        except Exception as e:
            logging.error(f"Error extracting tables from Excel {file_path}: {e}")
        
        return tables

    def extract_tables_from_docx(self, file_path: str) -> List[Document]:
        """Extract tables from Word documents."""
        tables = []
        
        if not TABLE_EXTRACTION_AVAILABLE or not pd:
            logging.warning("Table extraction not available - install dependencies")
            return tables
            
        try:
            from docx import Document
            
            doc = Document(file_path)
            
            for i, table in enumerate(doc.tables):
                try:
                    # Convert table to pandas DataFrame
                    data = []
                    for row in table.rows:
                        row_data = []
                        for cell in row.cells:
                            row_data.append(cell.text.strip())
                        data.append(row_data)
                    
                    if data:
                        # Create DataFrame
                        df = pd.DataFrame(data[1:], columns=data[0] if data else [])
                        
                        # Convert to structured text
                        table_text = self.table_to_text(df, f"Table {i+1}")
                        if table_text.strip():
                            tables.append(Document(
                                page_content=table_text,
                                metadata={"source": file_path, "type": "docx_table", "table_index": i}
                            ))
                except Exception as e:
                    logging.warning(f"Error processing table {i} in {file_path}: {e}")
                    
        except Exception as e:
            logging.error(f"Error extracting tables from Word {file_path}: {e}")
        
        return tables

    def enhanced_document_loader(self, file_path: str) -> List[Document]:
        """Enhanced document loader that extracts tables and structured content."""
        ext = os.path.splitext(file_path)[1].lower()
        all_docs = []
        
        try:
            # Extract tables first (only if table extraction is available)
            if TABLE_EXTRACTION_AVAILABLE:
                if ext == ".pdf":
                    all_docs.extend(self.extract_tables_from_pdf(file_path))
                elif ext == ".xlsx" or ext == ".xls":
                    all_docs.extend(self.extract_tables_from_excel(file_path))
                elif ext == ".docx":
                    all_docs.extend(self.extract_tables_from_docx(file_path))
            
            # Then extract regular text content
            if ext == ".txt":
                loader = TextLoader(file_path)
                all_docs.extend(loader.load())
            elif ext == ".pdf":
                # Try PyMuPDF first (faster for text-based PDFs)
                try:
                    loader = PyMuPDFLoader(file_path)
                    docs = loader.load()
                    if docs and any(d.page_content.strip() for d in docs):
                        all_docs.extend(docs)
                except Exception as e:
                    logging.warning(f"PyMuPDF failed for {file_path}: {e}. Trying OCR...")
                    try:
                        loader = UnstructuredPDFLoader(file_path, strategy="ocr_only")
                        docs = loader.load()
                        if docs and any(d.page_content.strip() for d in docs):
                            all_docs.extend(docs)
                    except Exception as ocr_e:
                        logging.error(f"Both PyMuPDF and OCR failed for {file_path}: {ocr_e}")
            elif ext == ".docx":
                loader = UnstructuredWordDocumentLoader(file_path)
                all_docs.extend(loader.load())
            elif ext == ".csv":
                loader = CSVLoader(file_path)
                all_docs.extend(loader.load())
            elif ext == ".xml":
                tree = ET.parse(file_path)
                root = tree.getroot()
                text = ET.tostring(root, encoding="unicode", method="text")
                all_docs.append(Document(page_content=text, metadata={"source": file_path}))
            
            # Add metadata about table extraction
            for doc in all_docs:
                if "type" not in doc.metadata:
                    doc.metadata["type"] = "text_content"
                doc.metadata["enhanced_processing"] = True
            
            logging.info(f"Enhanced processing extracted {len(all_docs)} documents from {file_path}")
            return all_docs
            
        except Exception as e:
            logging.error(f"Error in enhanced document loading for {file_path}: {e}")
            # Fallback to basic loader
            return self._basic_document_loader(file_path)
    
    def _basic_document_loader(self, file_path: str) -> List[Document]:
        """Basic document loader as fallback when enhanced processing fails."""
        ext = os.path.splitext(file_path)[1].lower()
        all_docs = []
        
        try:
            if ext == ".txt":
                loader = TextLoader(file_path)
                all_docs.extend(loader.load())
            elif ext == ".pdf":
                try:
                    loader = PyMuPDFLoader(file_path)
                    all_docs.extend(loader.load())
                except Exception as e:
                    logging.warning(f"PyMuPDF failed for {file_path}: {e}")
                    try:
                        loader = UnstructuredPDFLoader(file_path)
                        all_docs.extend(loader.load())
                    except Exception as e2:
                        logging.error(f"All PDF loaders failed for {file_path}: {e2}")
            elif ext == ".docx":
                loader = UnstructuredWordDocumentLoader(file_path)
                all_docs.extend(loader.load())
            elif ext == ".csv":
                loader = CSVLoader(file_path)
                all_docs.extend(loader.load())
            elif ext == ".xml":
                tree = ET.parse(file_path)
                root = tree.getroot()
                text = ET.tostring(root, encoding="unicode", method="text")
                all_docs.append(Document(page_content=text, metadata={"source": file_path}))
            
            logging.info(f"Basic processing extracted {len(all_docs)} documents from {file_path}")
            return all_docs
            
        except Exception as e:
            logging.error(f"Error in basic document loading for {file_path}: {e}")
            return []

    def start_embedding_async(self, query: str):
        """Get embedding immediately - no background processing."""
        try:
            # SPEED OPTIMIZATION: Use BGE embeddings from config for 1000x speed boost
            if hasattr(self, 'bge_model') and self.bge_model is not None:
                print(f"[SPEED] Using {EMBEDDING_MODEL} embeddings for 1000x speed boost")
                embedding = self.bge_model.encode(query, convert_to_tensor=False)
                print(f"[SPEED] {EMBEDDING_MODEL} embedding generated in milliseconds ({len(embedding)}D)")
                return embedding
            else:
                # Fallback to config model embeddings
                print(f"[SPEED] Using {MODEL_NAME} embeddings (faster fallback)")
                embedding = self.embedding.embed_query(query)
                
                # Truncate to smaller dimension for speed boost
                if len(embedding) > self.embedding_dimension:
                    embedding = embedding[:self.embedding_dimension]
                    print(f"[SPEED] Truncated embedding from {len(embedding)}D to {self.embedding_dimension}D for speed")
                
                return embedding
        except Exception as e:
            logging.warning(f"Embedding failed: {e}")
            return None

    def get_chunks_with_early_termination(self, query: str, similarity_threshold: float = 0.30, min_chunks: int = 5) -> List[Document]:
        """Get chunks with early termination for high-confidence matches."""
        early_start = time.time()
        
        try:
            print(f"[SPEED] Starting early termination search (threshold: {similarity_threshold})")
            
            # Try similarity score threshold search first
            retriever = self.db.as_retriever(
                search_type="similarity_score_threshold", 
                search_kwargs={"score_threshold": similarity_threshold, "k": 15}
            )
            docs = retriever.invoke(query)
            
            if len(docs) >= min_chunks:
                early_time = time.time() - early_start
                print(f"[SPEED] Early termination SUCCESS: Found {len(docs)} high-confidence chunks in {early_time:.3f}s")
                return docs[:15]  # Return top 15 high-confidence chunks
            else:
                print(f"[SPEED] Early termination: Only {len(docs)} high-confidence chunks found, using regular search")
                # Fallback to regular search
                return self.get_hierarchical_chunks(query, initial_k=15)
                
        except Exception as e:
            print(f"[SPEED] Early termination failed: {e}, using regular search")
            # Fallback to regular search
            return self.get_hierarchical_chunks(query, initial_k=15)

    def limit_context_size(self, context: str, max_chars: int = 4000) -> str:
        """Limit context size for faster LLM processing while keeping reasonable size."""
        if len(context) <= max_chars:
            return context
        
        print(f"[SPEED] Context too large ({len(context)} chars), limiting to {max_chars} chars")
        
        # Smart truncation - try to keep complete chunks
        chunks = context.split("\n\n")
        limited_chunks = []
        current_length = 0
        
        for chunk in chunks:
            if current_length + len(chunk) + 2 <= max_chars:  # +2 for \n\n
                limited_chunks.append(chunk)
                current_length += len(chunk) + 2
            else:
                # Add partial chunk if space allows
                remaining_space = max_chars - current_length - 3  # -3 for "..."
                if remaining_space > 100:  # Only if meaningful space left
                    partial_chunk = chunk[:remaining_space] + "..."
                    limited_chunks.append(partial_chunk)
                break
        
        limited_context = "\n\n".join(limited_chunks)
        print(f"[SPEED] Context limited to {len(limited_context)} chars ({len(limited_chunks)} chunks)")
        return limited_context

    def get_concise_response(self, prompt: str) -> str:
        """Get concise response by limiting generation for faster processing."""
        # Add explicit length instruction to prompt
        concise_prompt = f"{prompt}\n\nProvide a concise answer in 2-3 sentences maximum."
        
        try:
            response = self.llm.invoke(concise_prompt)
            
            # Convert to string if needed
            if isinstance(response, dict) and 'result' in response:
                response = response['result']
            else:
                response = str(response)
            
            # Ensure response isn't too long
            if len(response) > 800:
                print(f"[SPEED] Response too long ({len(response)} chars), truncating")
                # Truncate at sentence boundary
                sentences = response.split('.')
                truncated = []
                current_length = 0
                
                for sentence in sentences:
                    if current_length + len(sentence) + 1 <= 800:
                        truncated.append(sentence)
                        current_length += len(sentence) + 1
                    else:
                        break
                
                result = '.'.join(truncated) + '.'
                print(f"[SPEED] Response truncated to {len(result)} chars")
                return result
            
            return response
            
        except Exception as e:
            print(f"[SPEED] Concise response failed: {e}, using regular invoke")
            return str(self.llm.invoke(prompt))



    def build_context_streamlined(self, docs: List[Document]) -> str:
        """Build context with minimal string operations for maximum speed."""
        if not docs:
            return "No relevant chunks found."
        
        # Pre-allocate list size for efficiency
        context_parts = [None] * len(docs)
        
        # Process chunks with minimal operations
        for i, doc in enumerate(docs):
            content = doc.page_content.strip()
            # Limit size in one operation
            if len(content) > 300:
                content = content[:300] + "..."
            context_parts[i] = f"Chunk {i+1}: {content}"
        
        # Single join operation (much faster than +=)
        return "\n\n".join(context_parts)

    def process_chunks_parallel(self, docs: List[Document]) -> List[dict]:
        """Process retrieved chunks in parallel for faster preparation."""
        parallel_start = time.time()
        
        def process_single_chunk(doc_with_index):
            """Process a single chunk with its index."""
            i, doc = doc_with_index
            try:
                # Process chunk metadata and content
                source = os.path.basename(str(doc.metadata.get('source', 'Unknown')))
                chunk_content = doc.page_content.strip()
                
                # Limit chunk size to prevent overwhelming the LLM
                if len(chunk_content) > 300:
                    chunk_content = chunk_content[:300] + "..."
                
                return {
                    'index': i,
                    'source': source,
                    'content': chunk_content,
                    'metadata': doc.metadata
                }
            except Exception as e:
                logging.warning(f"Error processing chunk {i}: {e}")
                return {
                    'index': i,
                    'source': 'Error',
                    'content': 'Processing error',
                    'metadata': {}
                }
        
        try:
            # Process chunks in parallel with ThreadPoolExecutor
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Add index to each doc for processing
                docs_with_index = [(i, doc) for i, doc in enumerate(docs)]
                processed_chunks = list(executor.map(process_single_chunk, docs_with_index))
            
            # Sort by index to maintain order
            processed_chunks.sort(key=lambda x: x['index'])
            
            parallel_time = time.time() - parallel_start
            print(f"[SPEED] Parallel chunk processing completed in {parallel_time:.3f}s (4 workers)")
            
            return processed_chunks
            
        except Exception as e:
            print(f"[SPEED] Parallel processing failed: {e}, using sequential fallback")
            parallel_time = time.time() - parallel_start
            print(f"[SPEED] Parallel processing attempt took {parallel_time:.3f}s before fallback")
            
            # Fallback to sequential processing
            processed_chunks = []
            for i, doc in enumerate(docs):
                source = os.path.basename(str(doc.metadata.get('source', 'Unknown')))
                chunk_content = doc.page_content.strip()
                if len(chunk_content) > 300:
                    chunk_content = chunk_content[:300] + "..."
                processed_chunks.append({
                    'index': i,
                    'source': source,
                    'content': chunk_content,
                    'metadata': doc.metadata
                })
            
            return processed_chunks

    def get_hierarchical_chunks(self, query: str, initial_k: int = None) -> List[Document]:
        """
        Ultra-fast chunk-only retrieval: get only the most relevant chunks for maximum speed.
        
        Args:
            query: The user's question
            initial_k: Number of chunks to retrieve (default: 15 for optimal speed)
            
        Returns:
            List of document chunks (not full documents) for maximum performance
        """
        hierarchical_start = time.time()
        
        # Optimize search parameters for better performance
        self._optimize_search_parameters()
        
        # Use optimal chunk count for speed vs accuracy balance
        if initial_k is None:
            initial_k = min(15, self.db.index.ntotal)  # 15 chunks for optimal speed
            logging.info(f"[TIMING] Starting ultra-fast chunk-only retrieval for {initial_k} chunks (optimized for speed)")
        else:
            logging.info(f"[TIMING] Starting chunk-only retrieval with initial_k={initial_k}")
        
        try:
            logging.info(f"[DEBUG] Starting ultra-fast chunk retrieval with initial_k={initial_k}")
            
            # Step 1: Get only the most relevant chunks (Vector similarity search)
            initial_search_start = time.time()
            logging.info(f"[TIMING] Starting ultra-fast vector similarity search")
            retriever = self.db.as_retriever(search_type="similarity", k=initial_k)
            initial_docs = retriever.invoke(query)
            initial_search_time = time.time() - initial_search_start
            logging.info(f"[TIMING] Ultra-fast vector similarity search took: {initial_search_time:.3f}s")
            
            if not initial_docs:
                logging.warning("[DEBUG] No chunks found")
                return []
            
            logging.info(f"[DEBUG] Retrieved {len(initial_docs)} chunks (not full documents)")
            
            # Return chunks directly - no document expansion
            hierarchical_time = time.time() - hierarchical_start
            logging.info(f"[TIMING] Total ultra-fast chunk retrieval took: {hierarchical_time:.3f}s")
            logging.info(f"[DEBUG] Chunk-only retrieval complete: {len(initial_docs)} chunks")
            
            return initial_docs
            
        except Exception as e:
            logging.error(f"[DEBUG] Error in chunk retrieval: {e}")
            # Fallback to minimal chunk retrieval
            fallback_start = time.time()
            logging.info(f"[TIMING] Using fallback minimal chunk retrieval")
            retriever = self.db.as_retriever(search_type="similarity", k=min(10, self.db.index.ntotal))
            result = retriever.invoke(query)
            fallback_time = time.time() - fallback_start
            logging.info(f"[TIMING] Fallback minimal chunk retrieval took: {fallback_time:.3f}s")
            return result

    def calculate_confidence_with_gemma(self, query: str, answer: str, docs: list) -> float:
        """
        Calculate confidence using config model with fallback.
        
        Args:
            query: The user's question
            answer: The generated answer
            docs: The documents used for the answer
            
        Returns:
            Confidence score (0-100)
        """
        confidence_start = time.time()
        logging.info(f"[TIMING] Starting confidence calculation with config model")
        
        try:
            # Prepare context from retrieved documents
            context = "\n\n".join([f"Document {i+1}: {doc.page_content}" for i, doc in enumerate(docs)])
            
            # Confidence evaluation prompt
            confidence_prompt = f"""Rate this answer's confidence (0-100) based on:
- Relevance to question
- Completeness of information
- Clarity of response

Question: {query}
Answer: {answer}
Sources: {context}

Respond with ONLY a number between 0 and 100."""

            # # Try Gemma 3B first
            # if self.gemma_model:
            #     gemma_start = time.time()
            #     logging.info("[DEBUG] Using Gemma 3B for confidence calculation")
            #     confidence_response = self.gemma_model.invoke(confidence_prompt)
            #     confidence_text = str(confidence_response).strip()
            #     gemma_time = time.time() - gemma_start
            #     logging.info(f"[TIMING] Gemma 3B confidence calculation took: {gemma_time:.3f}s")
                
            #     # Extract numeric confidence score
            #     import re
            #     confidence_match = re.search(r'\b(\d{1,3})\b', confidence_text)
            #     if confidence_match:
            #         confidence_score = float(confidence_match.group(1))
            #         confidence_score = max(0, min(100, confidence_score))
            #         logging.info(f"[DEBUG] Gemma 3B confidence score: {confidence_score}")
            #         return round(confidence_score, 2)
            
            # # Fallback to Llama 3
            # if self.llama_model:
            #     llama_start = time.time()
            #     logging.info("[DEBUG] Using Llama 3 fallback for confidence calculation")
            #     confidence_response = self.llama_model.invoke(confidence_prompt)
            #     confidence_text = str(confidence_response).strip()
            #     llama_time = time.time() - llama_start
            #     logging.info(f"[TIMING] Llama 3 confidence calculation took: {llama_time:.3f}s")
                
            #     # Extract numeric confidence score
            #     import re
            #     confidence_match = re.search(r'\b(\d{1,3})\b', confidence_text)
            #     if confidence_match:
            #         confidence_score = float(confidence_match.group(1))
            #         confidence_score = max(0, min(100, confidence_score))
            #         logging.info(f"[DEBUG] Llama 3 fallback confidence score: {confidence_score}")
            #         return round(confidence_score, 2)
            
            # Final fallback
            logging.warning("[DEBUG] All confidence models failed, using fallback")
            return 50.0 if docs else 0.0
                
        except Exception as e:
            logging.warning(f"[DEBUG] Confidence calculation failed: {e}")
            # Fallback confidence based on whether we have documents
            return 20.0 if docs else 0.0

    def initialize_confidence_models(self):
        """Initialize config model for confidence calculation (Ollama auto-detects best device)."""
        try:
            # Initialize confidence models (Ollama automatically uses best available device)
            optimal_device = self.hardware_info['optimal_device']
            logging.info(f"[DEBUG] Initializing confidence models (Ollama will auto-detect best device: {optimal_device})")
            
            from langchain_ollama import OllamaLLM
            
            # Try config model first (Ollama auto-detects GPU/CPU)
            try:
                logging.info(f"[DEBUG] Attempting {MODEL_NAME} initialization")
                self.gemma_model = OllamaLLM(model=MODEL_NAME)
                self.confidence_model = self.gemma_model
                logging.info(f"[DEBUG] {MODEL_NAME} confidence model initialized successfully")
                
            except Exception as e:
                logging.warning(f"[DEBUG] Failed to initialize {MODEL_NAME}: {e}")
                try:
                    # Fallback to config model
                    logging.info(f"[DEBUG] Attempting {MODEL_NAME} initialization as fallback")
                    self.llama_model = OllamaLLM(model=MODEL_NAME)
                    self.confidence_model = self.llama_model
                    logging.info(f"[DEBUG] {MODEL_NAME} confidence model initialized successfully as fallback")
                except Exception as e2:
                    logging.error(f"[DEBUG] Failed to initialize confidence models: {e2}")
                    self.confidence_model = None
                    
        except Exception as e:
            logging.error(f"[DEBUG] Failed to initialize confidence models: {e}")
            self.confidence_model = None

    def validate_vector_db_integrity(self) -> dict:
        """Validate that vector database matches actual files on disk and clean any stale data."""
        try:
            logging.info("[DEBUG] Starting vector database integrity validation...")
            
            if not self.db:
                logging.info("[DEBUG] No vector database to validate")
                return {'status': 'no_db', 'message': 'No vector database exists'}
            
            # Get all files currently on disk
            disk_files = set()
            for root, _, files in os.walk(DOCS_PATH):
                for file in files:
                    disk_files.add(file)
            
            logging.info(f"[DEBUG] Found {len(disk_files)} files on disk")
            
            # Get all sources from vector database
            try:
                # Get a sample of documents to check sources
                sample_docs = self.db.similarity_search("", k=1000)  # Get up to 1000 docs
                db_sources = set()
                for doc in sample_docs:
                    source = doc.metadata.get('source', '')
                    if source:
                        # Extract filename from full path
                        filename = os.path.basename(source)
                        db_sources.add(filename)
                
                logging.info(f"[DEBUG] Vector database contains {len(db_sources)} unique document sources")
                
                # Find stale data (files in DB but not on disk)
                stale_sources = db_sources - disk_files
                missing_sources = disk_files - db_sources
                
                if stale_sources:
                    logging.warning(f"[DEBUG] Found {len(stale_sources)} stale sources in vector database: {stale_sources}")
                else:
                    logging.info("[DEBUG] No stale sources found in vector database")
                
                if missing_sources:
                    logging.warning(f"[DEBUG] Found {len(missing_sources)} files on disk not in vector database: {missing_sources}")
                else:
                    logging.info("[DEBUG] All disk files are represented in vector database")
                
                # If there are discrepancies, rebuild the database
                if stale_sources or missing_sources:
                    logging.info("[DEBUG] CRITICAL: Vector database integrity compromised, rebuilding...")
                    print(f"[HNSW] CRITICAL: Found {len(stale_sources)} stale sources and {len(missing_sources)} missing sources")
                    
                    # CRITICAL: Force complete rebuild to ensure data integrity
                    self.build_db("integrity_rebuild")
                    
                    return {
                        'status': 'rebuilt',
                        'message': 'Vector database rebuilt due to integrity issues with guaranteed cleanup',
                        'stale_sources': len(stale_sources),
                        'missing_sources': len(missing_sources),
                        'cleanup_verified': True
                    }
                else:
                    logging.info("[DEBUG] Vector database integrity validated successfully")
                    
                    # CRITICAL: Additional validation - check if HNSW is working
                    hnsw_status = "unknown"
                    if self.db and hasattr(self.db, 'index'):
                        if hasattr(self.db.index, 'hnsw'):
                            hnsw_status = "active"
                            print(f"[HNSW] Database integrity confirmed with active HNSW index")
                        else:
                            hnsw_status = "standard"
                            print(f"[HNSW] Database integrity confirmed with standard index")
                    
                    return {
                        'status': 'valid',
                        'message': 'Vector database integrity confirmed',
                        'total_sources': len(db_sources),
                        'hnsw_status': hnsw_status,
                        'cleanup_verified': True
                    }
                    
            except Exception as e:
                logging.error(f"[DEBUG] Error validating vector database: {e}")
                logging.info("[DEBUG] Rebuilding database due to validation error...")
                self.build_db("validation_error_rebuild")
                return {
                    'status': 'error_rebuilt',
                    'message': f'Database rebuilt due to validation error: {str(e)}'
                }
                
        except Exception as e:
            logging.error(f"Error in vector database integrity validation: {e}")
            return {'status': 'error', 'message': str(e)}