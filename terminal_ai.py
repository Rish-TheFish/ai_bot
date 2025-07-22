# import os
# import sys
# import csv
# import tempfile
# import threading
# import sounddevice as sd
# import numpy as np
# import wave
# import time
# import subprocess
# from datetime import datetime
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_ollama import OllamaLLM
# from langchain.chains import RetrievalQA
# from langchain_core.documents import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from PyPDF2 import PdfReader
# from docx import Document as DocxReader
# from config_details import DOCS_PATH, DB_PATH, EMBEDDING_MODEL, UPLOAD_PIN
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# if not any("ffmpeg" in os.path.basename(p).lower() for p in os.environ["PATH"].split(os.pathsep)):
#     ffmpeg_path = os.path.join(os.getcwd(), "ffmpeg.exe")
#     if os.path.exists(ffmpeg_path):
#         os.environ["PATH"] += os.pathsep + os.getcwd()

# import whisper

# # Whisper
# whisper_model = whisper.load_model("base")

# # Globals
# recording = False
# db = None
# embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
# llm = OllamaLLM(model="mistral")

# # S3 state
# s3 = None
# BUCKET_NAME = ""
# REGION_NAME = ""
# USE_S3 = False

# # Misuse tracking
# inappropriate_count = 0

# # Audio constants
# channels = 1
# fs = 16000
# sample_format = 'int16'

# # PIN state
# pin_verified = False

# def init_s3(bucket, region):
#     import boto3
#     global s3, BUCKET_NAME, REGION_NAME, USE_S3
#     s3 = boto3.client("s3", region_name=region)
#     BUCKET_NAME = bucket
#     REGION_NAME = region
#     USE_S3 = True

# def prompt_for_pin():
#     global pin_verified
#     if pin_verified:
#         return True
#     entered = input("🔒 Enter PIN to upload documents: ").strip()
#     if entered == UPLOAD_PIN:
#         print("✅ PIN verified. You may now upload documents.")
#         pin_verified = True
#         return True
#     else:
#         print("❌ Incorrect PIN.")
#         return False

# def evaluate_content_safety(text, type="input"):
#     try:
#         prompt = f"Determine if the following {type} is appropriate for a workplace compliance assistant. Respond only with 'yes' or 'no'.\n{text}"
#         response = llm.invoke(prompt).strip().lower()
#         return response.startswith("yes")
#     except Exception as e:
#         print(f"⚠️ Guardrail evaluation failed: {e}")
#         return True

# def load_db():
#     global db
#     try:
#         db = FAISS.load_local(DB_PATH, embedding, allow_dangerous_deserialization=True)
#         print("✅ Vector DB loaded.")
#     except:
#         print("⚠️ Vector DB not found. Please upload compliance documents.")

# def build_db_from_s3():
#     if not USE_S3:
#         print("❌ S3 not enabled.")
#         return
#     os.makedirs(DOCS_PATH, exist_ok=True)
#     response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix="docs/")
#     docs = response.get("Contents", [])
#     for obj in docs:
#         key = obj["Key"]
#         if key.endswith("/"):
#             continue
#         local_path = os.path.join(DOCS_PATH, os.path.basename(key))
#         s3.download_file(BUCKET_NAME, key, local_path)
#         print(f"📥 {key} → {local_path}")
#     build_db()

# def build_db():
#     global db
#     all_docs = []
#     for file in os.listdir(DOCS_PATH):
#         path = os.path.join(DOCS_PATH, file)
#         loader = get_loader(path)
#         if loader:
#             try:
#                 all_docs.extend(loader.load())
#             except Exception as e:
#                 print(f"❌ Error loading {file}: {e}")
#     if not all_docs:
#         print("⚠️ No documents to index.")
#         return
#     chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(all_docs)
#     db = FAISS.from_documents(chunks, embedding)
#     db.save_local(DB_PATH)
#     print(f"✅ Vector DB built with {len(chunks)} chunks.")

# def get_loader(path):
#     ext = os.path.splitext(path)[-1].lower()
#     try:
#         if ext == ".txt":
#             from langchain_community.document_loaders import TextLoader
#             return TextLoader(path)
#         elif ext == ".pdf":
#             from langchain_community.document_loaders import PyMuPDFLoader
#             return PyMuPDFLoader(path)
#         elif ext == ".docx":
#             from langchain_community.document_loaders import UnstructuredWordDocumentLoader
#             return UnstructuredWordDocumentLoader(path)
#         elif ext == ".csv":
#             from langchain_community.document_loaders import CSVLoader
#             return CSVLoader(path)
#     except:
#         return None

# def ask_question(query):
#     global inappropriate_count
#     if not db:
#         print("❌ Load or build a DB first.")
#         return

#     if not evaluate_content_safety(query, "input"):
#         inappropriate_count += 1
#         print("🚫 Inappropriate question. Please try again.")
#         if inappropriate_count >= 3:
#             print("🔒 You have been locked out for repeated misuse.")
#             sys.exit()
#         return

#     retriever = db.as_retriever(search_type="similarity", k=3)
#     chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

#     docs = retriever.get_relevant_documents(query)
#     sources = ', '.join(sorted(set(os.path.basename(d.metadata.get("source", "N/A")) for d in docs)))
#     confidence = round(min(100, (100 * len(docs) / 3)), 2) if docs else 60
#     answer = chain.run(query)

#     if not evaluate_content_safety(answer, "output"):
#         print("🚫 The AI generated an inappropriate response. Please rephrase your question.")
#         return

#     print(f"\n🤖 AI: {answer}")
#     print(f"🧠 Confidence: {confidence}%")
#     print(f"📎 Source(s): {sources}\n")

# def upload_to_s3_manual():
#     if not prompt_for_pin():
#         return
#     file_path = input("📂 Enter full file path to upload: ").strip()
#     if not os.path.isfile(file_path):
#         print("❌ File not found.")
#         return
#     filename = os.path.basename(file_path)
#     try:
#         if USE_S3:
#             s3.upload_file(file_path, BUCKET_NAME, f"docs/{filename}")
#             print(f"✅ Uploaded {filename} to S3")
#             build_db_from_s3()
#         else:
#             dest = os.path.join(DOCS_PATH, filename)
#             os.makedirs(DOCS_PATH, exist_ok=True)
#             with open(file_path, "rb") as src, open(dest, "wb") as dst:
#                 dst.write(src.read())
#             print(f"✅ Saved locally: {dest}")
#             build_db()
#     except Exception as e:
#         print(f"❌ Upload failed: {e}")

# def export_chat_log():
#     content = input("💬 Enter chat log content to export: ")
#     filename = f"chat_log_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
#     with open(filename, "w", encoding="utf-8") as f:
#         f.write(content)
#     if USE_S3:
#         s3.upload_file(filename, BUCKET_NAME, f"exports/{filename}")
#         print(f"✅ Chat log exported and uploaded to S3 as {filename}")
#     else:
#         print(f"✅ Chat log saved locally as {filename}")

# def record_audio_loop():
#     global recording
#     print("🎙️ Speak now. Say 'cease' or type 'cease' to stop.")

#     while recording:
#         audio = sd.rec(int(3 * fs), samplerate=fs, channels=channels, dtype=sample_format)
#         sd.wait()
#         audio_path = os.path.join(tempfile.gettempdir(), f"record_{int(time.time())}.wav")
#         with wave.open(audio_path, 'wb') as wf:
#             wf.setnchannels(channels)
#             wf.setsampwidth(2)
#             wf.setframerate(fs)
#             wf.writeframes(audio.tobytes())

#         try:
#             result = whisper_model.transcribe(audio_path)
#             query = result['text'].strip()
#             if query:
#                 print(f"🗣️ Transcribed: {query}")
#                 if "cease" in query.lower():
#                     query = query.lower().replace("cease", "").strip()
#                     recording = False
#                 if query:
#                     ask_question(query)
#         except Exception as e:
#             print(f"❌ Transcription error: {e}")

#         time.sleep(0.5)

# def run_terminal_bot(use_s3=False, bucket=None, region=None):
#     global recording
#     if use_s3:
#         init_s3(bucket, region)
#     load_db()

#     print("🧠 AI Compliance Assistant (Terminal Mode)")
#     print("Type commands: `upload`, `build`, `ask`, `voice`, `export`, `cease`, `gui`, `exit`")

#     while True:
#         cmd = input("👉 Command: ").strip().lower()

#         if cmd == "upload":
#             upload_to_s3_manual()
#         elif cmd == "build":
#             build_db_from_s3() if USE_S3 else build_db()
#         elif cmd == "ask":
#             q = input("❓ Your question: ")
#             ask_question(q)
#         elif cmd == "voice":
#             recording = True
#             threading.Thread(target=record_audio_loop).start()
#         elif cmd == "cease":
#             recording = False
#         elif cmd == "export":
#             export_chat_log()
#         elif cmd == "gui":
#             print("🔄 Switching to GUI mode...")
#             subprocess.run([sys.executable, "main.py"])
#             break
#         elif cmd == "exit":
#             print("👋 Exiting.")
#             break
#         else:
#             print("❌ Unknown command.")