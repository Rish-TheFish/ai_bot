import os, csv, shutil, threading, boto3, wave, re
from datetime import datetime
from config_details import DOCS_PATH, DB_PATH, EMBEDDING_MODEL, UPLOAD_PIN
from langchain_community.document_loaders import (
    TextLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader, CSVLoader
)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import xml.etree.ElementTree as ET
from PyPDF2 import PdfReader
from docx import Document as DocxReader
import pyaudio
# import terminal_ai
import config_details
import random
import hashlib
import psycopg2
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

if not any("ffmpeg" in os.path.basename(p).lower() for p in os.environ["PATH"].split(os.pathsep)):
    ffmpeg_path = os.path.join(os.getcwd(), "ffmpeg.exe")
    if os.path.exists(ffmpeg_path):
        os.environ["PATH"] += os.pathsep + os.getcwd()

import whisper

class AIApp:
    def __init__(self, master=None, bucket_name=None, region_name=None, use_s3=None):
        # self.master = master
        # if master:
        #     self.master.title("🧠 AI Compliance Assistant")
        #     self.master.geometry("1000x720")
        self.theme = "light"
        self.mode = "Q&A"
        self.embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
        self.llm = OllamaLLM(model="mistral")
        self.db = None
        self.recording = False
        self.locked_out = False
        self.inappropriate_count = 0
        self.pin_verified = False
        self.upload_button = None
        self.whisper_model = whisper.load_model("base")
        self.DOCS_PATH = DOCS_PATH  # Make it accessible for Flask

        if use_s3 is None:
            # answer = messagebox.askyesno("S3 Storage", "Do you want to use AWS S3 for storage?")
            # self.use_s3 = answer
            self.use_s3 = False  # Default to False for web interface
        else:
            self.use_s3 = use_s3

        if self.use_s3:
            # self.bucket = simpledialog.askstring("S3 Bucket", "Enter your AWS S3 Bucket Name:")
            # self.region = simpledialog.askstring("S3 Region", "Enter your AWS Region (e.g., us-east-1):")
            self.bucket = bucket_name
            self.region = region_name
            self.s3 = boto3.client("s3", region_name=self.region)
            config_details.BUCKET_NAME = self.bucket
            config_details.REGION_NAME = self.region
        else:
            self.bucket = None
            self.region = None
            self.s3 = None

        self.upload_enabled = False

        # if master:
        #     self.setup_layout()
        #     self.toggle_theme()
        self.load_vector_db()

    def setup_layout(self):
        # Web version: no-op
        pass

    def verify_upload_access(self):
        # Web version: no-op
        print("verify_upload_access called (stub)")
        return True

    def secure_upload_docs(self):
        # Web version: no-op
        print("secure_upload_docs called (stub)")
        return True
    

    def switch_to_terminal(self):
        # Web version: no-op
        print("switch_to_terminal called (stub)")
        return True
        


# This function toggles the theme
    def toggle_theme(self):
        # Web version: no-op
        pass


# This function evaluates the content's safety and appropriateness
    def evaluate_content_safety(self, text, type="input"):
        try:
            prompt = f"""Determine if the following {type} is appropriate for a workplace compliance assistant. 
            ONLY flag content that is clearly inappropriate for a professional workplace environment.
            Allow normal business questions, compliance inquiries, and professional discussions.
            Content that is off topic or not related to compliance should be allowed, unless it is clearly inappropriate.
            Only block content that contains: explicit sexual content, violence, hate speech, illegal activities, or threats.
            
            Respond only with 'yes' or 'no'.
            
            {text}"""
            response = self.llm.invoke(prompt).strip().lower()
            return response.startswith("yes")
        except Exception as e:
            # If safety check fails, allow the content (fail open)
            return True



# This function loads the vector database
    def load_vector_db(self):
        try:
            self.db = FAISS.load_local(DB_PATH, self.embedding, allow_dangerous_deserialization=True)
            # Count existing documents
            doc_count = len([f for f in os.listdir(DOCS_PATH) if os.path.isfile(os.path.join(DOCS_PATH, f))])
            print(f"📦 Vector database loaded successfully with {doc_count} existing documents.")
            return True
        except Exception as e:
            print(f"⚠️ Vector database not found or corrupted: {e}")
            return False



# This function gets the document status
    def get_document_status(self):
        """Get information about existing documents and database status"""
        try:
            # Check if database exists
            db_exists = os.path.exists(DB_PATH) and os.path.exists(os.path.join(DB_PATH, "index.faiss"))
            
            # Count documents
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
            return {
                "database_loaded": False,
                "database_exists": False,
                "document_count": 0,
                "documents": [],
                "error": str(e)
            }



# This function exports the chat content to a file
    def export_chat(self, chat_content):
        """Export chat content to file"""
        if not chat_content:
            return False
        filename = f"chat_log_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(chat_content)
        return True


# This function uploads the documents to the database
    def upload_docs(self, file_paths):
        """Upload documents from file paths"""
        if not file_paths:
            return False
        uploaded_files = []
        for path in file_paths:
            filename = os.path.basename(path)
            dest = os.path.join(DOCS_PATH, filename)
            if os.path.abspath(path) == os.path.abspath(dest):
                uploaded_files.append(filename)
                print(f"File {filename} already in correct location")
            else:
                try:
                    shutil.copy(path, dest)
                    uploaded_files.append(filename)
                    print(f"Copied {filename} to {dest}")
                except Exception as e:
                    print(f"Error copying {filename}: {e}")
                    continue
        if uploaded_files:
            self.build_db()
        return uploaded_files


# This function builds the database
    def build_db(self):
        all_docs = []
        for root, _, files in os.walk(DOCS_PATH):
            for file in files:
                loader = self.get_loader(os.path.join(root, file))
                if loader:
                    try:
                        if isinstance(loader, list):
                            all_docs.extend(loader)
                        else:
                            all_docs.extend(loader.load())
                    except Exception as e:
                        print(f"❌ Error loading {file}: {e}")
        if not all_docs:
            print("[DEBUG] No documents found to index.")
            return
        
        # Optimized text splitting for FAQ documents
        # Smaller chunks for FAQ documents to capture individual Q&A pairs better
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,  # Smaller chunks for FAQ documents
            chunk_overlap=100,  # More overlap to maintain context
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]  # Better splitting for FAQ format
        )
        chunks = text_splitter.split_documents(all_docs)
        print(f"[DEBUG] Number of chunks created: {len(chunks)}")
        self.db = FAISS.from_documents(chunks, self.embedding)
        self.db.save_local(DB_PATH)
        # Debug: print all unique sources in the chunks
        sources = set()
        for chunk in chunks:
            src = chunk.metadata.get('source')
            if src:
                sources.add(src)
        print(f"[DEBUG] Sources in vector DB: {sorted(sources)}")


# This function loads the documents into the database
    def get_loader(self, path):
        ext = os.path.splitext(path)[-1].lower()
        try:
            if ext == ".txt":
                # Improved: Only treat as chat/email if >60% of first 50 lines match chat/email patterns
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
        except:
            return None

    def process_chat_email_file(self, path):
        """Process chat histories and email threads to extract key information"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Detect if it's a chat/email format
            if self.is_chat_email_format(content):
                return self.extract_chat_email_knowledge(content, path)
            else:
                # Regular text file
                return TextLoader(path)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            return TextLoader(path)

    def is_chat_email_format(self, content):
        """Detect if content is chat or email format"""
        # Common patterns in chat/email files
        chat_patterns = [
            r'\d{1,2}:\d{2}',  # Time stamps
            r'\[.*?\]',  # Bracketed timestamps
            r'<.*?@.*?>',  # Email addresses
            r'From:|To:|Subject:',  # Email headers
            r'User:|Admin:|Support:',  # Chat participants
            r'^\w+:\s',  # Name: message format
        ]
        
        lines = content.split('\n')
        chat_line_count = 0
        total_lines = min(len(lines), 50)  # Check first 50 lines
        
        for line in lines[:total_lines]:
            for pattern in chat_patterns:
                if re.search(pattern, line):
                    chat_line_count += 1
                    break
        
        # If more than 30% of lines match chat patterns, consider it chat/email
        return (chat_line_count / total_lines) > 0.3

    def is_chat_email_format_strict(self, content):
        """Stricter detection: Only treat as chat/email if >60% of first 50 lines match chat/email patterns"""
        chat_patterns = [
            r'\d{1,2}:\d{2}',  # Time stamps
            r'\[.*?\]',  # Bracketed timestamps
            r'<.*?@.*?>',  # Email addresses
            r'From:|To:|Subject:',  # Email headers
            r'User:|Admin:|Support:',  # Chat participants
            r'^\w+:\s',  # Name: message format
        ]
        lines = content.split('\n')
        chat_line_count = 0
        total_lines = min(len(lines), 50)
        for line in lines[:total_lines]:
            for pattern in chat_patterns:
                if re.search(pattern, line):
                    chat_line_count += 1
                    break
        # Only treat as chat/email if >60% of lines match
        return (chat_line_count / total_lines) > 0.6

    def extract_chat_email_knowledge(self, content, source_path):
        """Extract key information from chat/email content and create FAQ-like documents"""
        try:
            # First, try to structure the content better
            structured_content = self.structure_chat_email_content(content)
            
            # Use Mistral to analyze the content and extract key information
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
{structured_content[:8000]}  # Limit to first 8000 chars to avoid token limits

Please provide a structured summary:"""

            # Get the analysis from Mistral
            analysis = self.llm.invoke(analysis_prompt)
            
            # Create a document with the extracted knowledge
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
            print(f"Error extracting knowledge from {source_path}: {e}")
            # Fallback to regular text processing
            return TextLoader(source_path)

    def structure_chat_email_content(self, content):
        """Structure chat/email content for better analysis"""
        lines = content.split('\n')
        structured_lines = []
        
        for line in lines:
            # Clean up common chat/email patterns
            line = line.strip()
            if not line:
                continue
                
            # Remove timestamps and formatting
            line = re.sub(r'\[\d{1,2}:\d{2}(:\d{2})?\]', '', line)
            line = re.sub(r'<\d{1,2}:\d{2}(:\d{2})?>', '', line)
            
            # Keep meaningful content
            if len(line) > 10:  # Only keep substantial lines
                structured_lines.append(line)
        
        return '\n'.join(structured_lines)

    def get_doc_topic_map(self):
        """Return a mapping: filename -> set of topic_ids"""
        DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:postGreSQL12#@localhost:5432/chat_history')
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
        return mapping


# This function handles inappropriate questions and all other non-standard inputs
    def handle_question(self, query, chat_history=None, topic_ids=None):
        """Handle a question with content safety checks and optional chat history context, with topic filtering"""
        if self.locked_out:
            return {"error": "You have been locked out for repeated misuse."}

        if not self.db:
            return {"error": "Please upload compliance documents first."}

        if not query or not query.strip():
            return {"error": "Question is required."}

        # Content safety check
        if not self.evaluate_content_safety(query, "input"):
            self.inappropriate_count += 1
            if self.inappropriate_count >= 3:
                self.locked_out = True
                return {"error": "Locked out for repeated misuse."}
            return {"error": "Inappropriate question. Please try again."}

        try:
            # REMOVE topic-based filtering: always use all docs
            filter_filenames = None
            answer, confidence, sources = self.query_answer(query, chat_history=chat_history, filter_filenames=filter_filenames)

            # STRICT FALLBACK: If answer is the strict fallback, return as such
            if answer == "No information found in the provided documents.":
                return {
                    "answer": answer,
                    "confidence": 0,
                    "sources": "N/A"
                }

            # DISABLED: Output content safety check for testing
            # if not self.evaluate_content_safety(answer, "output"):
            #     return {"error": "The AI generated an inappropriate response. Please rephrase your question."}

            return {
                "answer": answer,
                "confidence": confidence,
                "sources": sources
            }
        except Exception as e:
            return {"error": str(e)}



# This is the main function that answers the question
    def query_answer(self, query, chat_history=None, filter_filenames=None):
        if not self.db:
            raise Exception("Database not loaded. Please upload documents first.")
        
        # Prepend chat history to the query for context
        if chat_history:
            history_str = "\n".join([f"User: {item['question']}\nAI: {item['answer']}" for item in chat_history])
            full_query = f"Previous conversation:\n{history_str}\n\nCurrent question: {query}"
        else:
            full_query = query
        
        # Enhanced FAQ prompt for better FAQ document handling
        faq_prompt = f"""You are a helpful AI assistant that answers questions based on FAQ documents and compliance information.

IMPORTANT INSTRUCTIONS:
1. Answer questions based ONLY on the provided document content
2. If the information is in the documents, provide a clear, accurate answer
3. If the information is not in the documents, say \"I don't have information about that in the available documents\"
4. For FAQ documents, look for the most relevant Q&A pairs
5. Be concise but thorough in your responses
6. If multiple documents contain relevant information, synthesize the information clearly

User Question: {full_query}

Please provide an answer based on the document content:"""
        
        retriever = self.db.as_retriever(search_type="similarity", k=5)
        chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=retriever)

        # Use retriever.invoke to get relevant docs
        docs = retriever.invoke(full_query)
        # If filtering by filenames, filter the docs here
        if filter_filenames is not None:
            docs = [d for d in docs if os.path.basename(str(d.metadata.get("source", ""))) in filter_filenames]
        # STRICT FALLBACK: If no docs after filtering, return strict message
        if not docs:
            return ("No information found in the provided documents.", 0, "N/A")
        seen = set()
        sources = []
        top_similarity = None
        
        # Try to get similarity if possible
        try:
            # Get the embedding for the query
            query_emb = self.embedding.embed_query(full_query)
            # Get the FAISS index and all doc embeddings
            index = self.db.index
            doc_embeddings = index.reconstruct_n(0, index.ntotal)
            # Compute cosine similarity with all docs
            import numpy as np
            from numpy.linalg import norm
            similarities = []
            for i in range(index.ntotal):
                doc_emb = doc_embeddings[i]
                sim = np.dot(query_emb, doc_emb) / (norm(query_emb) * norm(doc_emb) + 1e-8)
                similarities.append(sim)
            # Find the max similarity for the top doc
            if similarities:
                top_similarity = max(similarities)
        except Exception as e:
            top_similarity = None
            
        for d in docs:
            src = os.path.basename(str(d.metadata.get("source", "N/A")))
            if src not in seen and src != "N/A":
                seen.add(src)
                sources.append(src)
                
        # Realistic confidence: map cosine similarity to 0-100%
        if top_similarity is not None:
            # Cosine similarity is in [-1, 1], map to [0, 100]
            confidence = round(((top_similarity + 1) / 2) * 100, 2)
        elif docs:
            confidence = 20.0  # fallback if docs found but no similarity
        else:
            confidence = 0.0  # no docs, no confidence
            
        # Use the user's actual query for retrieval and answer generation
        result = chain.invoke({'query': full_query})
        if isinstance(result, dict) and 'result' in result:
            answer = result['result']
        else:
            answer = str(result)
        if isinstance(answer, list):
            answer = "\n".join(str(a) for a in answer)
        return answer.strip(), confidence, ', '.join(sources) if sources else "N/A"

    def upload_questions(self, file_paths=None):
        print("upload_questions called (stub)")
        return []

    def upload_checker_docs(self, *args, **kwargs):
        print("upload_checker_docs called (stub)")
        return []

    def force_rebuild_db(self):
        """Delete the FAISS index files and rebuild the vector DB from scratch."""
        try:
            if os.path.exists(DB_PATH):
                shutil.rmtree(DB_PATH)
            print("[DEBUG] Deleted old FAISS index directory.")
        except Exception as e:
            print(f"[DEBUG] Error deleting FAISS index: {e}")
        self.build_db()
        print("[DEBUG] Forced rebuild of vector DB complete.")


# This function handles inappropriate questions and all other non-standard inputs
    # def toggle_recording(self):
    #     self.recording = not self.recording
    #     if self.recording:
    #         threading.Thread(target=self.record_and_transcribe).start()


    # def record_and_transcribe(self):
    #     filename = f"recording_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
    #     audio_path = os.path.join(".", filename)
    #     chunk = 1024
    #     sample_format = pyaudio.paInt16
    #     channels = 1
    #     fs = 16000
    #     seconds = 5
    #     p = pyaudio.PyAudio()
    #     stream = p.open(format=sample_format, channels=channels, rate=fs, input=True, frames_per_buffer=chunk)
    #     frames = []

    #     # self.chat_log("🎙️ Recording started... Speak now")

    #     for _ in range(0, int(fs / chunk * seconds)):
    #         if not self.recording:
    #             break
    #         data = stream.read(chunk)
    #         frames.append(data)

    #     stream.stop_stream()
    #     stream.close()
    #     p.terminate()

    #     wf = wave.open(audio_path, 'wb')
    #     wf.setnchannels(channels)
    #     wf.setsampwidth(p.get_sample_size(sample_format))
    #     wf.setframerate(fs)
    #     wf.writeframes(b''.join(frames))
    #     wf.close()

    #     # ✅ Upload safely to S3 only if properly configured
    #     if self.use_s3 and self.bucket:
    #         try:
    #             # self.s3.upload_file(audio_path, self.bucket, f"recordings/{filename}")
    #             pass # Web version: no-op
    #         except Exception as e:
    #             # self.chat_log(f"❌ Failed to upload audio to S3: {e}")
    #             pass # Web version: no-op

    #     # ✅ Transcribe and process
    #     try:
    #         result = self.whisper_model.transcribe(audio_path)
    #         query = result['text'].strip()
    #         if query:
    #             # self.chat_log(f"🎧 Transcribed: {query}")

    #             if not self.evaluate_content_safety(query, "input"):
    #                 self.inappropriate_count += 1
    #                 # self.chat_log("🚫 Inappropriate voice input. Please try again.")
    #                 if self.inappropriate_count >= 3:
    #                     # self.chat_log("🔒 Locked out for repeated misuse.")
    #                     self.locked_out = True
    #                 return

    #             answer, confidence, source = self.query_answer(query)

    #             if not self.evaluate_content_safety(answer, "output"):
    #                 # self.chat_log("🚫 The AI generated an inappropriate response. Please rephrase your question.")
    #                 return

    #             # self.chat_log(f"🤖 AI: {answer}\n🧠 Confidence: {confidence}%\n📎 Source: {source}")
    #     except Exception as e:
    #         # self.chat_log(f"❌ Audio processing error: {e}")
    #         pass
