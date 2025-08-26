from flask import Flask, render_template, request, jsonify, session, send_file
import os
from ai_bot import AIApp
from Logistics_Files.config_details import DOCS_PATH, DB_PATH, EMBEDDING_MODEL, MODEL_NAME, POSTGRES_PASSWORD
# UPLOAD_PIN commented out for now
from Logistics_Files.backend_log import add_backend_log, backend_logs
import io
import csv

# Handle PDF imports with fallbacks
PDF_READER_AVAILABLE = False
PdfReader = None

try:
    from PyPDF2 import PdfReader
    PDF_READER_AVAILABLE = True
    print("✓ PyPDF2 imported successfully")
except ImportError:
    try:
        from pypdf import PdfReader
        PDF_READER_AVAILABLE = True
        print("✓ pypdf imported successfully (PyPDF2 alternative)")
    except ImportError:
        try:
            import fitz  # PyMuPDF
            PDF_READER_AVAILABLE = True
            print("✓ PyMuPDF imported successfully (PDF fallback)")
        except ImportError:
            PDF_READER_AVAILABLE = False
            print("⚠ Warning: No PDF reader available. PDF uploads will not work.")

# Ensure PdfReader is available globally
if not PDF_READER_AVAILABLE:
    print("⚠ PDF processing will be disabled - no PDF reader available")

from docx import Document as DocxReader
import xml.etree.ElementTree as ET
import psycopg2
import psycopg2.extras
import re
import json
from datetime import datetime, timedelta
import logging
from werkzeug.utils import secure_filename
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import uuid

# Centralized allowed file extensions
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.csv', '.xml', '.xlsx'}

# Test log entry on startup
add_backend_log("TEST LOG: Flask app started and logging is configured.")

app = Flask(__name__)
# Set secret key from environment variable or use a fixed key for shared user system
app.secret_key = os.environ.get('SECRET_KEY') or 'shared_user_secret_key_2024'

# AI App with lazy initialization and background building
ai_app = None
ai_app_initializing = False
ai_app_ready = False
ai_app_error = None

# Session security settings
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FLASK_ENV') == 'production'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)  # Longer sessions for shared users

# Make sessions permanent by default for shared user system
@app.before_request
def make_session_permanent():
    session.permanent = True

# Database setup - PostgreSQL
DATABASE_URL = f'postgresql://postgres:{POSTGRES_PASSWORD}@localhost:5432/chat_history'

# Use environment variable for upload PIN
# UPLOAD_PIN = os.getenv('UPLOAD_PIN', '1964')  # Commented out for now

# Global progress tracking for questionnaire processing
questionnaire_progress = {}

def cleanup_old_progress():
    """Clean up old progress entries to prevent memory leaks"""
    global questionnaire_progress
    current_time = time.time()
    to_remove = []
    
    for job_id, progress in questionnaire_progress.items():
        # Remove entries older than 1 hour
        if 'timestamp' not in progress:
            progress['timestamp'] = current_time
        elif current_time - progress['timestamp'] > 3600:  # 1 hour
            to_remove.append(job_id)
    
    for job_id in to_remove:
        del questionnaire_progress[job_id]
        add_backend_log(f"Cleaned up old progress entry: {job_id}")

# Clean up old progress entries every hour
def start_progress_cleanup():
    """Start background cleanup of old progress entries"""
    def cleanup_loop():
        while True:
            try:
                cleanup_old_progress()
                time.sleep(3600)  # Run every hour
            except Exception as e:
                add_backend_log(f"Error in progress cleanup: {e}")
                time.sleep(3600)  # Continue even if there's an error
    
    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()

def get_db():
    """Get database connection with retry logic"""
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            return psycopg2.connect(DATABASE_URL)
        except psycopg2.OperationalError as e:
            if attempt < max_retries - 1:
                print(f"⚠ Database connection attempt {attempt + 1} failed: {e}")
                print(f"⚠ Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"❌ Database connection failed after {max_retries} attempts: {e}")
                raise e
        except Exception as e:
            print(f"❌ Unexpected database error: {e}")
            raise e

def init_db():
    """Initialize the database with tables"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(255) UNIQUE NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create default shared user if it doesn't exist
    cursor.execute('''
        INSERT INTO users (username, email, password_hash) 
        VALUES ('shared_user', 'shared@faqbot.com', 'default_hash')
        ON CONFLICT (username) DO NOTHING
    ''')
    
    # Create chat_history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,
            user_id INTEGER DEFAULT 1,
            session_id VARCHAR(255),
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            confidence REAL,
            sources TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create topics table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS topics (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) UNIQUE NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create documents table (without topic relationship)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255) UNIQUE NOT NULL,
            file_path VARCHAR(500) NOT NULL,
            file_size BIGINT,
            file_type VARCHAR(20),
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create document_topics junction table for many-to-many relationship
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_topics (
            id SERIAL PRIMARY KEY,
            document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
            topic_id INTEGER REFERENCES topics(id) ON DELETE CASCADE,
            UNIQUE(document_id, topic_id)
        )
    ''')
    
    conn.commit()
    conn.close()

def get_user_id():
    """Get current user ID - always returns shared user ID"""
    return 1  # Always return the shared user ID

def get_session_id():
    """Get or create session ID for the shared user"""
    # Always use the same session ID for the shared user
    if 'shared_session_id' not in session:
        session['shared_session_id'] = 'shared_session_001'
    return session['shared_session_id']

def get_chat_history(limit=None, session_key=None):
    """Get chat history from database for the shared user"""
    conn = get_db()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    user_id = get_user_id()  # Always returns 1 (shared user)
    order = 'ASC'
    limit_clause = ''
    if limit is not None:
        limit_clause = f'LIMIT {int(limit)}'
    
    if session_key:
        # If session_key is provided, use it directly
        cursor.execute(f'''
            SELECT question, answer, sources FROM chat_history 
            WHERE session_id = %s AND user_id = %s
            ORDER BY timestamp {order} 
            {limit_clause}
        ''', (session_key, user_id))
    else:
        # Get all chat history for the shared user
        cursor.execute(f'''
            SELECT question, answer, sources FROM chat_history 
            WHERE user_id = %s 
            ORDER BY timestamp {order} 
            {limit_clause}
        ''', (user_id,))
    
    history = cursor.fetchall()
    conn.close()
    return [{
        'question': row['question'], 
        'answer': row['answer'],
        #'confidence': row['confidence'],
        'sources': row['sources']
    } for row in history]

def save_chat_history(question, answer, sources, session_key=None):
    """Save chat history to database for the shared user"""
    conn = get_db()
    cursor = conn.cursor()
    user_id = get_user_id()  # Always returns 1 (shared user)
    session_id = session_key if session_key else get_session_id()
    
    #try:
    #    confidence = float(confidence) if confidence is not None else None
    #except Exception:
    #    confidence = None
    
    # Handle structured source data
    if isinstance(sources, dict) and 'summary' in sources:
        # Convert structured sources to string format for database storage
        sources_str = sources.get('summary', 'N/A')
    else:
        # Legacy string format
        sources_str = str(sources) if sources else 'N/A'
    
    cursor.execute('''
        INSERT INTO chat_history (user_id, session_id, question, answer, sources)
        VALUES (%s, %s, %s, %s, %s)
    ''', (user_id, session_id, question, answer, sources_str))
    conn.commit()
    conn.close()

# Initialize database on startup (handle failures gracefully)
try:
    init_db()
    print("✓ Database initialized successfully")
except Exception as e:
    print(f"⚠ Database initialization failed: {e}")
    print("⚠ Flask will start without database - some features may not work")
    print("⚠ Database will be initialized when PostgreSQL becomes available")

def get_or_init_ai_app():
    """Get AI app instance with lazy initialization"""
    global ai_app, ai_app_initializing, ai_app_ready, ai_app_error
    
    # Ensure AIApp is available in this scope
    try:
        from ai_bot import AIApp
    except ImportError as e:
        ai_app_error = f"Failed to import AIApp: {e}"
        add_backend_log(f"Failed to import AIApp: {e}")
        return None
    
    # If already ready, return it
    if ai_app_ready and ai_app is not None:
        return ai_app
    
    # If there's an error, return None
    if ai_app_error:
        return None
    
    # If already initializing, wait a bit and return current state
    if ai_app_initializing:
        return ai_app
    
    # Quick check: if no documents exist, make AI app ready immediately
    if not os.path.exists(DOCS_PATH) or not any(os.listdir(DOCS_PATH)):
        add_backend_log("No documents found - making AI app ready immediately")
        try:
            ai_app = AIApp(None)
            ai_app_ready = True
            ai_app_error = None
            add_backend_log("AI app created successfully for empty documents")
        except Exception as e:
            add_backend_log(f"Error creating AI app: {e}")
            ai_app_ready = True
            ai_app_error = str(e)
        return ai_app
    
    # Quick check: if database already exists and is valid, try to load it immediately
    try:
        temp_ai_app = AIApp(None)
        if temp_ai_app.database_exists_and_valid():
            add_backend_log("Database exists - attempting quick load...")
            if temp_ai_app.load_vector_db() and temp_ai_app.db is not None:
                ai_app = temp_ai_app
                ai_app_ready = True
                ai_app_error = None
                add_backend_log("Database loaded successfully in quick initialization!")
                return ai_app
            else:
                add_backend_log("Quick load failed, will use background initialization")
        else:
            add_backend_log("No valid database found, will use background initialization")
    except Exception as e:
        add_backend_log(f"Quick initialization check failed: {e}, will use background initialization")
    
    # CRITICAL FIX: Make AI app ready immediately, then build database in background
    add_backend_log("Creating AI app instance for immediate use...")
    try:
        ai_app = AIApp(None, skip_db_init=True)  # Skip database init for immediate readiness
        # Mark as ready immediately so users can start asking questions
        ai_app_ready = True
        ai_app_error = None
        add_backend_log("AI app ready immediately! Database will build in background.")
        
        # Start database building in background (non-blocking)
        def build_db_background():
            global ai_app_error
            try:
                add_backend_log("Starting background database build...")
                if os.path.exists(DOCS_PATH) and any(os.listdir(DOCS_PATH)):
                    add_backend_log("Building vector database from documents in background...")
                    ai_app.build_db("background_build")
                    if ai_app.db is not None:
                        add_backend_log("Background database build completed successfully!")
                        ai_app_error = None
                    else:
                        ai_app_error = "Background database build failed"
                        add_backend_log("Background database build failed")
                else:
                    add_backend_log("No documents to process in background")
            except Exception as e:
                ai_app_error = f"Background database build error: {e}"
                add_backend_log(f"Background database build error: {e}")
        
        # Start background database building
        db_thread = threading.Thread(target=build_db_background, daemon=True)
        db_thread.start()
        
        return ai_app
        
    except Exception as e:
        ai_app_error = str(e)
        add_backend_log(f"Error creating AI app: {e}")
        # Fall back to background initialization
        ai_app_initializing = True
        add_backend_log("Falling back to background initialization...")
        
        def init_ai_background():
            global ai_app, ai_app_initializing, ai_app_ready, ai_app_error
            try:
                add_backend_log("Creating AI app instance in background...")
                ai_app = AIApp(None)
                add_backend_log("AI app instance created successfully in background")
                
                # Check if database exists and is valid
                add_backend_log("Checking if database exists and is valid...")
                try:
                    db_valid = ai_app.database_exists_and_valid()
                    add_backend_log(f"Database validation result: {db_valid}")
                    
                    if db_valid:
                        add_backend_log("Loading existing vector database...")
                        load_success = ai_app.load_vector_db()
                        if load_success and ai_app.db is not None:
                            ai_app_ready = True
                            ai_app_error = None
                            add_backend_log("AI app ready with existing database")
                        else:
                            add_backend_log("Database load failed, will rebuild...")
                            ai_app_ready = False
                    else:
                        # Check if there are documents to process
                        add_backend_log("Checking for documents to process...")
                        if os.path.exists(DOCS_PATH) and any(os.listdir(DOCS_PATH)):
                            add_backend_log("Building vector database from documents...")
                            ai_app.build_db("initial_build")
                            if ai_app.db is not None:
                                ai_app_ready = True
                                ai_app_error = None
                                add_backend_log("AI app ready with new database")
                            else:
                                ai_app_ready = False
                                ai_app_error = "Database build failed"
                        else:
                            add_backend_log("No documents found - AI app ready for uploads")
                            ai_app_ready = True
                            ai_app_error = None
                except Exception as db_error:
                    add_backend_log(f"Error during database validation/loading: {db_error}")
                    # Only set as ready if we actually have a working database
                    if ai_app.db is not None:
                        ai_app_ready = True
                        ai_app_error = None
                        add_backend_log("AI app ready despite database error")
                    else:
                        ai_app_ready = False
                        ai_app_error = f"Database error: {db_error}"
                        add_backend_log("AI app not ready - database failed to load")
                    
            except Exception as e:
                ai_app_error = str(e)
                add_backend_log(f"Error initializing AI app: {e}")
                # Only set as ready if we have a working database
                if ai_app and ai_app.db is not None:
                    ai_app_ready = True
                    add_backend_log("AI app ready despite initialization error")
                else:
                    ai_app_ready = False
                    add_backend_log("AI app not ready due to initialization error")
            finally:
                ai_app_initializing = False
                add_backend_log("Background initialization completed")
        
        # Start background thread
        thread = threading.Thread(target=init_ai_background, daemon=True)
        thread.start()
        
        # Set a timeout to prevent hanging
        def timeout_handler():
            global ai_app_initializing, ai_app_ready, ai_app_error
            time.sleep(300)  # 5 minute timeout for background initialization
            if ai_app_initializing:
                add_backend_log("Background initialization timed out after 5 minutes - checking database state")
                ai_app_initializing = False
                
                # Check if we actually have a working database before setting as ready
                if ai_app and ai_app.db is not None:
                    ai_app_ready = True
                    ai_app_error = None
                    add_backend_log("Timeout occurred but database is loaded - setting as ready")
                else:
                    ai_app_ready = False
                    ai_app_error = "Background initialization timed out - database not loaded"
                    add_backend_log("Timeout occurred and database not loaded - app not ready")
        
        timeout_thread = threading.Thread(target=timeout_handler, daemon=True)
        timeout_thread.start()
    
    return ai_app

def ensure_ai_app_ready():
    """Ensure AI app is ready, building database if needed"""
    global ai_app, ai_app_ready
    
    if not ai_app_ready:
        ai_app = get_or_init_ai_app()
        if ai_app is None:
            return False
    
    # Additional check: ensure database is actually loaded
    if ai_app_ready and ai_app and ai_app.db is None:
        add_backend_log("AI app marked as ready but database not loaded - forcing refresh")
        try:
            if ai_app.database_exists_and_valid():
                ai_app.load_vector_db()
                if ai_app.db is not None:
                    add_backend_log("Database loaded successfully during refresh")
                else:
                    ai_app_ready = False
                    add_backend_log("Database load failed during refresh")
            else:
                ai_app_ready = False
                add_backend_log("Database validation failed during refresh")
        except Exception as e:
            ai_app_ready = False
            add_backend_log(f"Error during database refresh: {e}")
    
    return ai_app_ready

@app.route('/')
def index():
    # No guest session cleanup needed - all users are shared users
    
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Simple health check endpoint for Cloud Run"""
    try:
        # Basic health check - just return OK if Flask is running
        return jsonify({
            'status': 'healthy',
            'service': 'FAQ Bot',
            'timestamp': datetime.now().isoformat(),
            'flask': 'running'
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/get_database_status')
def get_database_status():
    """Simple PostgreSQL health check endpoint for frontend monitoring"""
    try:
        # Test PostgreSQL connection
        conn = get_db()
        cursor = conn.cursor()
        
        # Simple query to test connection
        cursor.execute('SELECT 1')
        cursor.fetchone()
        
        conn.close()
        
        return jsonify({
            'status': 'connected',
            'database': 'postgresql',
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        add_backend_log(f"Database health check failed: {e}")
        return jsonify({
            'status': 'disconnected',
            'database': 'postgresql',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/user_status', methods=['GET'])
def user_status():
    """Always return shared user status"""
    return jsonify({
        'logged_in': True,
        'username': 'shared_user',
        'user_id': 1
    })

@app.route('/clear_history', methods=['POST'])
def clear_history():
    conn = get_db()
    cursor = conn.cursor()
    
    user_id = get_user_id()  # Always returns 1 (shared user)
    
    cursor.execute('DELETE FROM chat_history WHERE user_id = %s', (user_id,))
    
    conn.commit()
    conn.close()
    return jsonify({'status': 'success', 'message': 'Chat history cleared'})

@app.route('/export_history', methods=['POST'])
def export_history():
    conn = get_db()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    user_id = get_user_id()  # Always returns 1 (shared user)
    
    cursor.execute('''
        SELECT question, answer, timestamp FROM chat_history 
        WHERE user_id = %s ORDER BY timestamp
    ''', (user_id,))
    
    history = cursor.fetchall()
    conn.close()
    
    # Create CSV
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Timestamp', 'Question', 'Answer'])
    for row in history:
        writer.writerow([row['timestamp'], row['question'], row['answer']])
    
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='chat_history.csv'
    )

@app.route('/init_ai', methods=['POST'])
def init_ai():
    global ai_app, ai_app_ready, ai_app_error
    
    try:
        # Use lazy initialization
        ai_app = get_or_init_ai_app()
        
        if ai_app_error and ai_app_ready:
            # App is ready but has a warning
            try:
                doc_status = ai_app.get_document_status()
            except:
                doc_status = {'database_loaded': False, 'document_count': 0}
            return jsonify({
                'status': 'success', 
                'message': f'AI Assistant ready (with warning: {ai_app_error})', 
                'document_status': doc_status,
                'ready': True,
                'warning': ai_app_error
            })
        elif ai_app_error and not ai_app_ready:
            return jsonify({'error': f'Failed to initialize AI Assistant: {ai_app_error}'})
        
        if ai_app_ready:
            try:
                doc_status = ai_app.get_document_status()
            except Exception as e:
                doc_status = {'database_loaded': False, 'document_count': 0, 'error': str(e)}
            return jsonify({
                'status': 'success', 
                'message': 'AI Assistant ready', 
                'document_status': doc_status,
                'ready': True
            })
        else:
            return jsonify({
                'status': 'initializing', 
                'message': 'AI Assistant is initializing in background...',
                'ready': False
            })
    except Exception as e:
        add_backend_log(f"Error in init_ai endpoint: {str(e)}")
        return jsonify({'error': f'Failed to initialize AI Assistant: {str(e)}'})

@app.route('/upload', methods=['POST'])
def upload_docs():
    global ai_app, ai_app_ready, ai_app_error
    
    if ai_app_error:
        return jsonify({'error': f'AI Assistant error: {ai_app_error}'})
    
    if not ai_app_ready:
        return jsonify({'error': 'AI Assistant not ready. Please wait for initialization to complete.'})
    
    if not ai_app:
        return jsonify({'error': 'AI Assistant not available'})
    
    # pin = request.form.get('pin')
    # if pin != UPLOAD_PIN:
    #     return jsonify({'error': 'Invalid security PIN.'})
    
    files = request.files.getlist('files')
    temp_paths = []
    for file in files:
        if file.filename:
            # Sanitize filename
            safe_filename = secure_filename(file.filename)
            temp_path = os.path.join(DOCS_PATH, safe_filename)
            file.save(temp_path)
            temp_paths.append(temp_path)
    
    result = ai_app.upload_docs(temp_paths)
    add_backend_log(f"Files uploaded: {', '.join([os.path.basename(f) for f in temp_paths])}")
    return jsonify({'status': 'success', 'files': result})

@app.route('/export_chat', methods=['POST'])
def export_chat():
    global ai_app
    if not ai_app:
        return jsonify({'error': 'AI Assistant not initialized'})
    data = request.get_json()
    chat_content = data.get('chat_content', '')
    result = ai_app.export_chat(chat_content)
    return jsonify({'status': 'success' if result else 'error'})

@app.route('/ai_status', methods=['GET'])
def get_ai_status():
    """Get the current status of the AI Assistant"""
    global ai_app, ai_app_ready, ai_app_initializing, ai_app_error
    
    if ai_app_error:
        return jsonify({
            'status': 'error',
            'message': f'AI Assistant error: {ai_app_error}',
            'ready': False
        })
    
    if not ai_app_ready:
        if ai_app_initializing:
            return jsonify({
                'status': 'initializing',
                'message': 'AI Assistant is initializing in background...',
                'ready': False
            })
        else:
            return jsonify({
                'status': 'not_started',
                'message': 'AI Assistant has not been started yet',
                'ready': False
            })
    
    if not ai_app:
        return jsonify({
            'status': 'not_available',
            'message': 'AI Assistant not available',
            'ready': False
        })
    
    return jsonify({
        'status': 'ready',
        'message': 'AI Assistant is ready',
        'ready': True,
        'has_database': ai_app.db is not None
    })

@app.route('/document_status', methods=['GET'])
def get_document_status():
    global ai_app, ai_app_ready, ai_app_initializing, ai_app_error
    
    if ai_app_error and ai_app_ready:
        # If there's an error but app is ready, return ready status with warning
        return jsonify({
            'status': 'ready_with_warning',
            'message': f'AI Assistant ready (with warning: {ai_app_error})',
            'ready': True,
            'warning': ai_app_error
        })
    elif ai_app_error and not ai_app_ready:
        return jsonify({'error': f'AI Assistant error: {ai_app_error}'})
    
    if not ai_app_ready:
        if ai_app_initializing:
            return jsonify({
                'status': 'initializing',
                'message': 'AI Assistant is initializing in background...',
                'ready': False
            })
        else:
            # Trigger initialization
            ai_app = get_or_init_ai_app()
            return jsonify({
                'status': 'starting',
                'message': 'Starting AI Assistant initialization...',
                'ready': False
            })
    
    if not ai_app:
        return jsonify({'error': 'AI Assistant not available'})
    
    # Force refresh AI app state if documents exist but database isn't loaded
    docs_exist = os.path.exists(DOCS_PATH) and any(os.listdir(DOCS_PATH))
    if docs_exist and not ai_app.db:
        add_backend_log("Documents exist but vector database not loaded - forcing refresh")
        try:
            if ai_app.database_exists_and_valid():
                load_success = ai_app.load_vector_db()
                if load_success and ai_app.db is not None:
                    ai_app_ready = True
                    add_backend_log("Vector database loaded successfully")
                else:
                    ai_app_ready = False
                    add_backend_log("Vector database load failed")
            else:
                ai_app.build_db("force_refresh")
                if ai_app.db is not None:
                    ai_app_ready = True
                    add_backend_log("Vector database rebuilt successfully")
                else:
                    ai_app_ready = False
                    add_backend_log("Vector database rebuild failed")
        except Exception as e:
            add_backend_log(f"Error refreshing vector database: {e}")
            ai_app_ready = False
    
    try:
        result = ai_app.get_document_status()
        result['ready'] = ai_app_ready
        result['database_loaded'] = ai_app.db is not None
        result['initializing'] = ai_app_initializing
        result['error'] = ai_app_error
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'status': 'ready_with_warning',
            'message': f'AI Assistant ready (with warning: {str(e)})',
            'ready': ai_app_ready,
            'warning': str(e),
            'database_loaded': ai_app.db is not None if ai_app else False,
            'initializing': ai_app_initializing,
            'error': ai_app_error
        })

@app.route('/ask', methods=['POST'])
def ask_question():
    global ai_app, ai_app_ready, ai_app_initializing, ai_app_error
    
    if ai_app_error:
        return jsonify({'error': f'AI Assistant error: {ai_app_error}'})
    
    if not ai_app_ready:
        if ai_app_initializing:
            return jsonify({'error': 'AI Assistant is still initializing. Please wait a moment and try again.'})
        elif ai_app_error:
            return jsonify({'error': f'AI Assistant error: {ai_app_error}'})
        else:
            # Trigger initialization
            try:
                ai_app = get_or_init_ai_app()
                if ai_app:
                    ai_app_ready = True
                    ai_app_error = None
                else:
                    return jsonify({'error': 'AI Assistant initialization failed. Please try again later.'})
            except Exception as e:
                ai_app_error = str(e)
                return jsonify({'error': f'AI Assistant initialization error: {e}'})
            return jsonify({'error': 'AI Assistant is starting up. Please wait a moment and try again.'})
    
    if not ai_app:
        return jsonify({'error': 'AI Assistant not available'})
    
    # Additional check: ensure database is actually loaded
    if ai_app.db is None:
        return jsonify({'error': 'AI Assistant database not loaded. Please wait for initialization to complete.'})
    
    data = request.get_json()
    question = data.get('question', '')
    topic_ids = data.get('topic_ids', ['none'])
    session_key = data.get('session_key', None)
    
    if not question:
        return jsonify({'error': 'Question is required'})
    
    # Security check removed - no longer needed with shared user system
    
    chat_history = get_chat_history(session_key=session_key)
    
    # Handle topic selection logic
    if 'all' in topic_ids:
        topic_ids = ['all']
    elif 'none' in topic_ids or not topic_ids:
        topic_ids = []
    # Otherwise, topic_ids is a list of selected topic IDs
    
    # COMPLETE ISOLATION - reset context for each individual question
    if ai_app:
        ai_app._reset_context_state()
    
    try:
        result = ai_app.handle_question(question, chat_history=chat_history, topic_ids=topic_ids)
        if 'error' in result:
            return jsonify({'error': result['error']})
        
        save_chat_history(question, result['answer'], result['sources'], session_key=session_key)
        return jsonify({
            'answer': result['answer'],
            #'confidence': result['confidence'],
            'sources': result['sources']
        })
    except Exception as e:
        ai_app_error = str(e)
        return jsonify({'error': f'AI Assistant error: {e}'})

@app.route('/update_document_topics', methods=['POST'])
def update_document_topics():
    """Update the topics associated with a document"""
    global ai_app
    
    if not ai_app:
        return jsonify({'error': 'AI Assistant not initialized'})
    
    data = request.get_json()
    filename = data.get('filename')
    topic_ids = data.get('topic_ids', [])
    
    if not filename:
        return jsonify({'error': 'Filename is required'})
    
    try:
        # Update document topics in the database
        result = ai_app.update_document_topics(filename, topic_ids)
        
        if result:
            add_backend_log(f"Updated topics for document '{filename}' to: {topic_ids}")
            
            # Get the topic names for the frontend
            topic_names = []
            if topic_ids:
                try:
                    conn = psycopg2.connect(DATABASE_URL)
                    cursor = conn.cursor()
                    cursor.execute('SELECT name FROM topics WHERE id = ANY(%s)', (topic_ids,))
                    topic_names = [row[0] for row in cursor.fetchall()]
                    conn.close()
                except Exception as e:
                    print(f"Error fetching topic names: {e}")
            
            return jsonify({
                'status': 'success', 
                'message': 'Document topics updated successfully',
                'updated_topics': topic_names
            })
        else:
            return jsonify({'error': 'Failed to update document topics'})
            
    except Exception as e:
        error_msg = f"Error updating document topics: {str(e)}"
        add_backend_log(error_msg)
        return jsonify({'error': error_msg})

@app.route('/upload_questions', methods=['POST'])
def upload_questions():
    """Upload and process questionnaire with parallel processing and timeouts"""
    global ai_app, questionnaire_progress
    
    if not ai_app:
        return jsonify({'error': 'AI Assistant not initialized'})
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    filename = file.filename
    if not filename:
        return jsonify({'error': 'No file selected'})
    
    # COMPLETE ISOLATION - clear any previous questionnaire progress to prevent contamination
    global questionnaire_progress
    questionnaire_progress.clear()  # Clear all previous progress
    
    # Generate job ID for progress tracking
    job_id = str(uuid.uuid4())
    questionnaire_progress[job_id] = {
        'status': 'starting',
        'total_questions': 0,
        'completed_questions': 0,
        'current_question': '',
        'message': 'Starting questionnaire processing...'
    }
    
    add_backend_log(f"Starting questionnaire processing for: {filename} (Job ID: {job_id})")
    
    # COMPLETE ISOLATION - reset AI app state before processing new questionnaire
    if ai_app:
        upload_session_id = str(uuid.uuid4())
        ai_app.set_upload_session(upload_session_id, job_id)
        add_backend_log(f"AI app state reset for new questionnaire session: {upload_session_id}")
        
        # Optionally create a completely fresh instance for maximum isolation
        # Uncomment the next line if you want a new instance for each questionnaire
        # ai_app = get_or_init_ai_app()  # This would create a completely new instance
    
    # Extract questions based on file type
    questions = extract_questions_from_file(file)
    
    if not questions:
        questionnaire_progress[job_id]['status'] = 'error'
        questionnaire_progress[job_id]['message'] = 'No questions found in file'
        return jsonify({'error': 'No questions found in file'})
    
    # Update progress
    questionnaire_progress[job_id].update({
        'status': 'processing',
        'total_questions': len(questions),
        'message': f'Found {len(questions)} questions, starting parallel processing...'
    })
    
    add_backend_log(f"Found {len(questions)} questions, starting parallel processing...")
    
    # Process questions in parallel with timeouts
    results = process_questions_parallel(questions, job_id)
    
    # Update progress to completed and store results
    questionnaire_progress[job_id].update({
        'status': 'completed',
        'completed_questions': len(results),
        'message': f'Completed processing {len(results)} questions',
        'results': results  # Store the results for later retrieval
    })
    
    add_backend_log(f"Completed processing {len(results)} questions")
    
    # Generate CSV and return results
    return generate_questionnaire_results(results, filename, job_id)

def extract_questions_from_file(file):
    """Extract questions from uploaded file"""
    ext = file.filename.split('.')[-1].lower()
    questions = []
    
    try:
        if ext == 'csv':
            stream = io.StringIO(file.stream.read().decode('utf-8'))
            reader = csv.reader(stream)
            for row in reader:
                if row and row[0].strip():
                    questions.append(row[0].strip())
        
        elif ext == 'txt':
            content = file.stream.read().decode('utf-8')
            questions = extract_questions_from_text(content)
        
        elif ext == 'pdf':
            if not PDF_READER_AVAILABLE:
                add_backend_log("PDF processing not available - no PDF reader installed")
                return []
            
            file.stream.seek(0)
            try:
                # Try PyMuPDF first (most reliable)
                try:
                    import fitz
                    pdf_bytes = file.stream.read()
                    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
                    full_text = ""
                    for page in pdf:
                        text = page.get_text()
                        if text:
                            full_text += text + "\n"
                    pdf.close()
                    questions = extract_questions_from_text(full_text)
                except ImportError:
                    # Fallback to PdfReader if available
                    if PdfReader:
                        pdf = PdfReader(io.BytesIO(file.stream.read()))
                        full_text = ""
                        for page in pdf.pages:
                            text = page.extract_text()
                            if text:
                                full_text += text + "\n"
                        questions = extract_questions_from_text(full_text)
                    else:
                        add_backend_log("No PDF reader available")
                        return []
            except Exception as pdf_error:
                add_backend_log(f"Error processing PDF: {pdf_error}")
                return []
        
        elif ext == 'docx':
            file.stream.seek(0)
            try:
                doc = DocxReader(io.BytesIO(file.stream.read()))
                for para in doc.paragraphs:
                    line = para.text.strip()
                    if line and len(line) > 10:
                        questions.append(line)
            except Exception as docx_error:
                add_backend_log(f"Error processing DOCX: {docx_error}")
                return []
        
        else:
            add_backend_log(f"Unsupported file type: {ext}")
            return []
            
    except Exception as e:
        add_backend_log(f"Error extracting questions: {e}")
        return []
    
    # Clean and validate questions
    cleaned_questions = []
    for q in questions:
        cleaned_q = clean_question_text(q)
        if cleaned_q and len(cleaned_q) > 10:
            cleaned_questions.append(cleaned_q)
    
    add_backend_log(f"Extracted {len(cleaned_questions)} valid questions from {len(questions)} raw questions")
    if len(cleaned_questions) == 0 and len(questions) > 0:
        add_backend_log("Warning: All questions were filtered out during cleaning. Check question length and content.")
    return cleaned_questions

def extract_questions_from_text(text):
    """Extract questions from text using multiple patterns"""
    question_patterns = [
        r'\n\s*\d+\)',  # 1), 2), etc.
        r'\n\s*Q\d+\.',  # Q1., Q2., etc.
        r'\n\s*Question\s+\d+:',  # Question 1:, Question 2:, etc.
        r'\n\s*\d+\.',  # 1., 2., etc.
        r'\n\s*\d+\s*[A-Z]',  # 1 A, 2 B, etc.
        r'\n\s*[A-Z]\)',  # A), B), etc.
        r'\n\s*[a-z]\)',  # a), b), etc.
        r'\n\s*[A-Z]\.',  # A., B., etc.
        r'\n\s*[a-z]\.',  # a., b., etc.
    ]
    
    questions = []
    if text.strip():
        # Try each pattern and see which one gives the most questions
        best_pattern = None
        best_questions = []
        
        for pattern in question_patterns:
            parts = re.split(pattern, text)
            if len(parts) > 1:
                pattern_questions = []
                for part in parts[1:]:
                    question = part.strip()
                    if question and len(question) > 10:
                        pattern_questions.append(question)
                
                if len(pattern_questions) > len(best_questions):
                    best_questions = pattern_questions
                    best_pattern = pattern
        
        if best_questions:
            questions = best_questions
            add_backend_log(f"Found {len(questions)} questions using pattern: {best_pattern}")
        else:
            # More flexible fallback: split on multiple delimiters
            add_backend_log("No structured patterns found, using flexible text splitting...")
            
            # Split on common question delimiters
            delimiters = [r'\n\s*\n', r'\n\s*•', r'\n\s*-\s*', r'\n\s*\*\s*', r'\n\s*→', r'\n\s*→']
            
            for delimiter in delimiters:
                parts = re.split(delimiter, text)
                if len(parts) > 1:
                    potential_questions = []
                    for part in parts:
                        question = part.strip()
                        if question and len(question) > 10:
                            # Check if it looks like a question
                            if any(word in question.lower() for word in ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'could', 'would', 'should', 'do', 'does', 'is', 'are', 'was', 'were']):
                                potential_questions.append(question)
                    
                    if len(potential_questions) > len(questions):
                        questions = potential_questions
                        add_backend_log(f"Found {len(questions)} questions using delimiter: {delimiter}")
            
            # If still no questions, try splitting on question marks
            if not questions:
                question_mark_parts = re.split(r'\?', text)
                for part in question_mark_parts[:-1]:
                    question = part.strip()
                    if question and len(question) > 10:
                        sentences = re.split(r'[.!?]', question)
                        if sentences:
                            last_sentence = sentences[-1].strip()
                            if last_sentence and len(last_sentence) > 10:
                                questions.append(last_sentence + "?")
                
                if questions:
                    add_backend_log(f"Found {len(questions)} questions using question mark splitting")
            
            # Final fallback: split on double newlines and look for question-like content
            if not questions:
                paragraphs = re.split(r'\n\s*\n', text)
                for para in paragraphs:
                    para = para.strip()
                    if para and len(para) > 15:  # Longer paragraphs are more likely to be questions
                        # Check if it contains question words or ends with question mark
                        if (any(word in para.lower() for word in ['what', 'how', 'why', 'when', 'where', 'who', 'which']) or 
                            para.endswith('?') or 
                            para.endswith('.')):
                            questions.append(para)
                
                if questions:
                    add_backend_log(f"Found {len(questions)} questions using paragraph splitting")
    
    return questions

def clean_question_text(question):
    """Clean and normalize question text"""
    # Remove question numbers at the beginning
    cleaned_q = re.sub(r'^\s*(?:\d+\)|Q\d+\.|Question\s+\d+:|\d+\.)\s*', '', question.strip())
    
    # Normalize to single line
    cleaned_q = re.sub(r'\s*\n\s*', ' ', cleaned_q)
    
    # Remove extra whitespace
    cleaned_q = re.sub(r'\s+', ' ', cleaned_q).strip()
    
    # Remove question marks at the beginning
    cleaned_q = re.sub(r'^\s*\?\s*', '', cleaned_q)
    
    # Ensure proper punctuation
    if cleaned_q and not cleaned_q.endswith(('.', '?', '!')):
        cleaned_q += '?'
    
    return cleaned_q

def process_questions_parallel(questions, job_id):
    """Process questions in parallel with timeouts"""
    results = []
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all questions for processing
        future_to_question = {
            executor.submit(process_single_question, q, i, job_id): (q, i) 
            for i, q in enumerate(questions)
        }
        
        # Collect results as they complete (with timeout)
        # 72 questions take ~70 seconds each, so allow 90 minutes total for safety
        for future in as_completed(future_to_question, timeout=5400):  # 90 minute total timeout
            question, index = future_to_question[future]
            try:
                result = future.result(timeout=180)  # 3 minutes per question timeout (allows for complex questions)
                results.append(result)
                add_backend_log(f"Completed question {index+1}/{len(questions)}: {question[:50]}...")
            except TimeoutError:
                add_backend_log(f"Question {index+1} timed out after 120 seconds")
                results.append({
                    'question': question,
                    'answer': 'Processing timed out - question was too complex or system was overloaded',
                    #'confidence': 0,
                    'sources': {'summary': 'Timeout occurred', 'detailed': 'Processing timed out'}
                })
            except Exception as e:
                add_backend_log(f"Error processing question {index+1}: {e}")
                results.append({
                    'question': question,
                    'answer': f'Error processing question: {str(e)}',
                    #'confidence': 0,
                    'sources': {'summary': 'Error occurred', 'detailed': str(e)}
                })
    
    # Sort results by original question order
    results.sort(key=lambda x: questions.index(x['question']))
    return results

def process_single_question(question, index, job_id):
    """Process a single question with timeout protection"""
    global ai_app, questionnaire_progress
    
    try:
        # Update progress
        if job_id in questionnaire_progress:
            questionnaire_progress[job_id].update({
                'current_question': f'Processing question {index+1}: {question[:50]}...',
                'message': f'Processing question {index+1} of {questionnaire_progress[job_id]["total_questions"]}'
            })
        
        add_backend_log(f"Processing question {index+1}: {question[:50]}...")
        
        # Check if AI app is ready
        if not ai_app or not hasattr(ai_app, 'db') or not ai_app.db:
            return {
                'question': question,
                'answer': 'AI Assistant not properly initialized. Please upload documents first.',
                #'confidence': 0,
                'sources': {'summary': 'AI not ready', 'detailed': 'AI Assistant not initialized'}
            }
        
        # COMPLETE ISOLATION - reset context for each individual question
        if ai_app:
            ai_app._reset_context_state()
        
        # Process the question using the EXACT same method as single questions
        # This ensures questionnaire questions get the same quality and speed as individual questions
        # Note: Questions are already cleaned in extract_questions_from_file()
        result = ai_app.handle_question(question)
        
        if 'error' in result:
            return {
                'question': question,
                'answer': f"Error: {result['error']}",
                #'confidence': 0,
                'sources': {'summary': 'Error occurred', 'detailed': result['error']}
            }
        
        # Extract answer and sources
        answer = result.get('answer', 'No answer generated')
        #confidence = result.get('confidence', 0)
        sources = result.get('sources', {})
        
        if isinstance(sources, dict):
            sources_data = sources
            sources_for_db = sources.get('summary', 'N/A')
        else:
            sources_data = {'summary': str(sources), 'detailed': str(sources)}
            sources_for_db = str(sources)
        
        # Save to database
        try:
            save_chat_history(question, answer, sources_for_db)
        except Exception as e:
            add_backend_log(f"Warning: Could not save question {index+1} to database: {e}")
        
        # Update progress
        if job_id in questionnaire_progress:
            questionnaire_progress[job_id]['completed_questions'] += 1
            questionnaire_progress[job_id]['message'] = f'Completed {questionnaire_progress[job_id]["completed_questions"]} of {questionnaire_progress[job_id]["total_questions"]} questions'
        
        return {
            'question': question,  # Keep original question for display
            'answer': answer,
            #'confidence': confidence,
            'sources': sources_data
        }
        
    except Exception as e:
        add_backend_log(f"Exception processing question {index+1}: {str(e)}")
        return {
            'question': question,
            'answer': f'Error processing question: {str(e)}',
            #'confidence': 0,
            'sources': {'summary': 'Error occurred', 'detailed': str(e)}
        }

def generate_questionnaire_results(results, filename, job_id):
    """Generate CSV and return questionnaire results"""
    try:
        # Generate CSV
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_ALL)
        writer.writerow(['Question', 'Answer', 'Source'])
        
        for row in results:
            try:
                sources_str = ""
                if isinstance(row['sources'], dict):
                    sources_str = row['sources'].get('summary', 'N/A')
                else:
                    sources_str = str(row['sources'])
                
                clean_answer = re.sub(r'\s+', ' ', str(row['answer'])).strip()
                
                question = str(row['question']).encode('utf-8', errors='ignore').decode('utf-8')
                answer = clean_answer.encode('utf-8', errors='ignore').decode('utf-8')
                source = sources_str.encode('utf-8', errors='ignore').decode('utf-8')
                #confidence = str(row['confidence'])
                
                writer.writerow([question, answer, source])
            except Exception as row_error:
                add_backend_log(f"Error writing CSV row: {row_error}")
                writer.writerow(['Error processing question', 'Error processing answer', 'N/A'])
        
        output.seek(0)
        
        # Create temporary file
        import tempfile
        import uuid
        
        unique_id = str(uuid.uuid4())
        csv_filename = f'answers_{len(results)}_questions_{unique_id}.csv'
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.write(output.getvalue())
        temp_file.close()
        
        # Store file mapping
        if not hasattr(app, 'temp_files'):
            app.temp_files = {}
        app.temp_files[unique_id] = temp_file.name
        
        add_backend_log(f"Questionnaire processing complete: {len(results)} questions processed")
        
        return jsonify({
            'questions': [r['question'] for r in results],
            'answers': [r['answer'] for r in results],
            #'confidences': [r['confidence'] for r in results],
            'sources': [r['sources'] for r in results],
            'csv_id': unique_id,
            'filename': csv_filename,
            'job_id': job_id
        })
        
    except Exception as e:
        add_backend_log(f"Error generating results: {e}")
        return jsonify({'error': f'Error generating results: {str(e)}'})

@app.route('/questionnaire_results/<job_id>', methods=['GET'])
def get_questionnaire_results(job_id):
    """Get the final results for a completed questionnaire job"""
    try:
        if job_id not in questionnaire_progress:
            return jsonify({'error': 'Job not found'}), 404
        
        progress = questionnaire_progress[job_id]
        if progress['status'] != 'completed':
            return jsonify({'error': 'Job not yet completed'}), 400
        
        # Return the results that were stored during processing
        if 'results' not in progress:
            return jsonify({'error': 'No results found for this job'}), 404
        
        results = progress['results']
        
        # Generate CSV info
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_ALL)
        writer.writerow(['Question', 'Answer', 'Source'])
        
        for row in results:
            try:
                sources_str = ""
                if isinstance(row['sources'], dict):
                    sources_str = row['sources'].get('summary', 'N/A')
                else:
                    sources_str = str(row['sources'])
                
                clean_answer = re.sub(r'\s+', ' ', str(row['answer'])).strip()
                
                question = str(row['question']).encode('utf-8', errors='ignore').decode('utf-8')
                answer = clean_answer.encode('utf-8', errors='ignore').decode('utf-8')
                source = sources_str.encode('utf-8', errors='ignore').decode('utf-8')
                #confidence = str(row['confidence'])
                
                writer.writerow([question, answer, source])
            except Exception as row_error:
                add_backend_log(f"Error writing CSV row: {row_error}")
                writer.writerow(['Error processing question', 'Error processing answer', 'N/A'])
        
        output.seek(0)
        
        # Create temporary file
        import tempfile
        import uuid
        
        unique_id = str(uuid.uuid4())
        csv_filename = f'answers_{len(results)}_questions_{unique_id}.csv'
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.write(output.getvalue())
        temp_file.close()
        
        # Store file mapping
        if not hasattr(app, 'temp_files'):
            app.temp_files = {}
        app.temp_files[unique_id] = temp_file.name
        
        return jsonify({
            'questions': [r['question'] for r in results],
            'answers': [r['answer'] for r in results],
            #'confidences': [r['confidence'] for r in results],
            'sources': [r['sources'] for r in results],
            'csv_id': unique_id,
            'filename': csv_filename,
            'job_id': job_id
        })
        
    except Exception as e:
        add_backend_log(f"Error getting questionnaire results: {e}")
        return jsonify({'error': f'Error getting results: {str(e)}'})

@app.route('/download_csv/<file_id>', methods=['GET'])
def download_csv(file_id):
    """Download the generated CSV file"""
    try:
        # Get the temp file path from the stored mapping
        if not hasattr(app, 'temp_files') or file_id not in app.temp_files:
            return jsonify({'error': 'File not found or expired'}), 404
        
        temp_file_path = app.temp_files[file_id]
        
        if os.path.exists(temp_file_path):
            try:
                return send_file(
                    temp_file_path,
                    mimetype='text/csv',
                    as_attachment=True,
                    download_name=f'answers_{file_id}.csv'
                )
            finally:
                # Clean up in a separate thread to avoid blocking
                import threading
                def cleanup_async():
                    try:
                        os.unlink(temp_file_path)
                        if hasattr(app, 'temp_files'):
                            app.temp_files.pop(file_id, None)
                    except:
                        pass
                
                cleanup_thread = threading.Thread(target=cleanup_async)
                cleanup_thread.daemon = True
                cleanup_thread.start()
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Simple rate limiting for get_history endpoint
history_request_times = {}

@app.route('/get_history', methods=['GET'])
def get_history():
    global history_request_times
    session_key = request.args.get('session_key')
    
    # Add debug logging
    add_backend_log(f"DEBUG: get_history called for session_key: {session_key}")
    
    # TEMPORARILY DISABLED: Rate limiting to get chat working again
    # TODO: Re-enable after frontend is properly fixed
    # current_time = time.time()
    # if session_key in history_request_times:
    #     time_since_last = current_time - history_request_times[session_key]
    #     if time_since_last < 0.5:  # 0.5 second cooldown
    #         add_backend_log(f"Rate limit hit for get_history: {session_key} (last request: {time_since_last:.2f}s ago)")
    #         return jsonify({'error': 'Rate limit exceeded. Please wait before requesting again.'}), 429
    
    # Update last request time
    history_request_times[session_key] = time.time()
    
    # Clean up old entries (older than 1 hour)
    cleanup_time = time.time() - 3600
    history_request_times = {k: v for k, v in history_request_times.items() if v > cleanup_time}
    
    # Get chat history
    history = get_chat_history(session_key=session_key)
    add_backend_log(f"DEBUG: get_history returning {len(history)} items for session {session_key}")
    
    return jsonify({'history': history})


@app.route('/get_documents', methods=['GET'])
def get_documents():
    """Get list of uploaded documents"""
    try:
        docs_dir = DOCS_PATH
        documents = []
        topic_id = request.args.get('topic', 'all')
        selected_topics_json = request.args.get('selected_topics')
        selected_topics = []
        if selected_topics_json:
            try:
                selected_topics = json.loads(selected_topics_json)
            except Exception:
                selected_topics = []
        # Parse sort parameter
        sort_param = request.args.get('sort', '')
        sort_columns = []
        allowed_sort = {'name': 'filename', 'type': 'file_type', 'uploaded': 'uploaded_at', 'topics': 'tags'}
        for part in sort_param.split(','):
            if ':' in part:
                col, direction = part.split(':', 1)
                col = col.strip()
                direction = direction.strip().lower()
                if col in allowed_sort and direction in ('asc', 'desc'):
                    sort_columns.append((allowed_sort[col], direction))
        # Default sort
        if not sort_columns:
            sort_columns = [('filename', 'asc')]
        # Build ORDER BY clause
        order_by = ', '.join([f"{col} {direction.upper()}" for col, direction in sort_columns if col != 'tags'])
        # Determine which files to show
        if topic_id == 'all':
            # Show all files
            filenames_to_show = [f for f in os.listdir(docs_dir) if os.path.isfile(os.path.join(docs_dir, f))]
        elif topic_id == 'multiple' and selected_topics:
            # Convert all topic IDs to integers
            selected_topics = [int(tid) for tid in selected_topics if str(tid).isdigit()]
            conn = get_db()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(f'''
                SELECT d.filename
                FROM documents d
                JOIN document_topics dt ON d.id = dt.document_id
                WHERE dt.topic_id = ANY(%s)
                GROUP BY d.id, d.filename
                HAVING COUNT(DISTINCT dt.topic_id) = %s
                {f'ORDER BY {order_by}' if order_by else ''}
            ''', (selected_topics, len(selected_topics)))
            topic_files = [row['filename'] for row in cursor.fetchall()]
            conn.close()
            filenames_to_show = topic_files
        else:
            # Show files that have this specific topic
            conn = get_db()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(f'''
                SELECT DISTINCT d.filename
                FROM documents d
                JOIN document_topics dt ON d.id = dt.document_id
                WHERE dt.topic_id = %s
                {f'ORDER BY {order_by}' if order_by else ''}
            ''', (topic_id,))
            topic_files = [row['filename'] for row in cursor.fetchall()]
            conn.close()
            filenames_to_show = topic_files
        # Get all tags for all files in one query
        if filenames_to_show:
            conn = get_db()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute('''
                SELECT d.filename, t.name as tag
                FROM documents d
                JOIN document_topics dt ON d.id = dt.document_id
                JOIN topics t ON dt.topic_id = t.id
                WHERE d.filename = ANY(%s)
                ORDER BY d.filename, t.name
            ''', (filenames_to_show,))
            tag_rows = cursor.fetchall()
            conn.close()
            # Map: filename -> [tag, tag, ...]
            file_tags = {}
            for row in tag_rows:
                file_tags.setdefault(row['filename'], []).append(row['tag'])
        else:
            file_tags = {}
        for filename in filenames_to_show:
            file_path = os.path.join(docs_dir, filename)
            if not os.path.isfile(file_path):
                continue
            file_size = os.path.getsize(file_path)
            file_time = os.path.getmtime(file_path)
            upload_date = datetime.fromtimestamp(file_time).strftime('%Y-%m-%d %H:%M')
            file_ext = filename.split('.')[-1].lower()
            file_type = file_ext.upper()
            tags = file_tags.get(filename, [])
            documents.append({
                'name': filename,
                'type': file_type,
                'size': file_size,
                'uploaded': upload_date,
                'tags': tags
            })
        # Sort by upload date (newest first) if not already sorted
        if not sort_columns or (sort_columns and sort_columns[0][0] != 'uploaded_at'):
            documents.sort(key=lambda x: x['uploaded'], reverse=True)
        return jsonify({'documents': documents})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/check_filename_conflict', methods=['POST'])
def check_filename_conflict():
    """Check if a filename already exists"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'Filename not provided'})
        
        docs_dir = DOCS_PATH
        file_path = os.path.join(docs_dir, filename)
        
        exists = os.path.exists(file_path)
        return jsonify({
            'exists': exists,
            'filename': filename
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/rename_document', methods=['POST'])
def rename_document():
    """Rename an existing document"""
    try:
        data = request.get_json()
        old_filename = data.get('old_filename')
        new_filename = data.get('new_filename')
        
        if not old_filename or not new_filename:
            return jsonify({'error': 'Both old and new filenames are required'})
        
        docs_dir = 'your_docs'
        old_file_path = os.path.join(docs_dir, old_filename)
        new_file_path = os.path.join(docs_dir, new_filename)
        
        if not os.path.exists(old_file_path):
            return jsonify({'error': 'Original file not found'})
        
        if os.path.exists(new_file_path):
            return jsonify({'error': 'Target filename already exists'})
        
        # Validate new filename extension
        new_file_ext = os.path.splitext(new_filename)[1].lower()
        if new_file_ext not in ALLOWED_EXTENSIONS:
            return jsonify({'error': f'File type {new_file_ext} not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'})
        
        # Rename the file
        os.rename(old_file_path, new_file_path)
        
        # Wait a moment to ensure file system operations complete
        import time
        time.sleep(0.5)
        
        # Always reinitialize AI with updated documents
        global ai_app
        ai_app = AIApp(None)
        ai_app.build_db()
        
        add_backend_log(f"Document renamed: {old_filename} to {new_filename}")
        return jsonify({'status': 'success', 'message': f'Document renamed from {old_filename} to {new_filename}'})
    except Exception as e:
        add_backend_log(f"Error message: {e}")
        return jsonify({'error': str(e)})

@app.route('/upload_document', methods=['POST'])
def upload_document():
    """Upload a new document with incremental database update"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        # Validate file type
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            return jsonify({'error': f'File type {file_ext} not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'})
        
        # Check for filename conflict
        docs_dir = DOCS_PATH
        os.makedirs(docs_dir, exist_ok=True)
        file_path = os.path.join(docs_dir, file.filename)
        
        if os.path.exists(file_path):
            return jsonify({
                'status': 'conflict',
                'message': f'File "{file.filename}" already exists. Please rename or cancel.',
                'filename': file.filename
            })
        
        # Save file
        file.save(file_path)
        
        # Get selected topics from frontend
        topic_ids_json = request.form.get('topic_ids', '[]')
        topic_ids = json.loads(topic_ids_json)
        # Always convert topic_ids to integers
        topic_ids = [int(tid) for tid in topic_ids if str(tid).isdigit()]
        
        # Save document info to database
        conn = get_db()
        cursor = conn.cursor()
        
        file_size = os.path.getsize(file_path)
        file_type = file_ext.upper()
        
        # Insert document record
        cursor.execute('''
            INSERT INTO documents (filename, file_path, file_size, file_type)
            VALUES (%s, %s, %s, %s) RETURNING id
        ''', (file.filename, file_path, file_size, file_type))
        
        result = cursor.fetchone()
        if result is None:
            raise Exception("Failed to insert document record")
        document_id = result[0]
        
        # Insert topic relationships if topics are specified
        if topic_ids and len(topic_ids) > 0:
            for topic_id in topic_ids:
                cursor.execute('''
                    INSERT INTO document_topics (document_id, topic_id)
                    VALUES (%s, %s)
                ''', (document_id, topic_id))
        
        conn.commit()
        conn.close()
        
        # Update AI app with new document
        global ai_app, ai_app_ready
        try:
            if ai_app and ai_app.db:
                # Database exists, use incremental update
                if ai_app.add_document_incremental(file_path):
                    add_backend_log(f"Document uploaded incrementally: {file.filename}")
                else:
                    # Fallback to full rebuild if incremental fails
                    add_backend_log(f"Incremental update failed, rebuilding database: {file.filename}")
                    ai_app.build_db("incremental_update")
                    add_backend_log(f"Document uploaded with fallback rebuild: {file.filename}")
            else:
                # No database exists, do initial build
                add_backend_log(f"Building initial database for: {file.filename}")
                ai_app = AIApp(None)
                ai_app.build_db("initial_build")
                add_backend_log(f"Document uploaded with initial build: {file.filename}")
            
            # Ensure AI app is ready after document addition
            ai_app_ready = True
            add_backend_log(f"AI app ready after document upload: {file.filename}")
            
        except Exception as e:
            add_backend_log(f"Error updating AI app: {e}")
            # Force rebuild as fallback
            try:
                ai_app = AIApp(None)
                ai_app.build_db("force_rebuild")
                ai_app_ready = True
                add_backend_log(f"AI app rebuilt after error: {file.filename}")
            except Exception as rebuild_error:
                add_backend_log(f"Failed to rebuild AI app: {rebuild_error}")
                ai_app_ready = False
        
        return jsonify({'status': 'success', 'message': f'Document {file.filename} uploaded successfully'})
    except Exception as e:
        add_backend_log(f"Error message: {e}")
        return jsonify({'error': str(e)})

@app.route('/replace_document', methods=['POST'])
def replace_document():
    """Replace an existing document with conflict resolution"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        old_filename = request.form.get('old_filename')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if not old_filename:
            return jsonify({'error': 'Original filename not provided'})
        
        # Validate file type
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            return jsonify({'error': f'File type {file_ext} not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'})
        
        # Check if old file exists
        docs_dir = 'your_docs'
        if old_filename is None:
            return jsonify({'error': 'Invalid old filename'})
        old_file_path = os.path.join(docs_dir, old_filename)
        if not os.path.exists(old_file_path):
            return jsonify({'error': 'Original file not found'})
        
        # Check for filename conflict (if new filename is different from old)
        if file.filename != old_filename:
            new_file_path = os.path.join(docs_dir, file.filename)
            if os.path.exists(new_file_path):
                return jsonify({
                    'status': 'conflict',
                    'message': f'File "{file.filename}" already exists. Please rename or cancel.',
                    'filename': file.filename,
                    'old_filename': old_filename
                })
        
        # Delete old file and save new one
        os.remove(old_file_path)
        new_file_path = os.path.join(docs_dir, file.filename)
        file.save(new_file_path)
        
        # Get selected topics from frontend
        topic_ids_json = request.form.get('topic_ids', '[]')
        topic_ids = json.loads(topic_ids_json)
        # Always convert topic_ids to integers
        topic_ids = [int(tid) for tid in topic_ids if str(tid).isdigit()]
        
        # Update document info in database
        conn = get_db()
        cursor = conn.cursor()
        
        file_size = os.path.getsize(new_file_path)
        file_type = file_ext.upper()
        
        # Get the document ID for the old filename
        cursor.execute('SELECT id FROM documents WHERE filename = %s', (old_filename,))
        result = cursor.fetchone()
        if result:
            document_id = result[0]
            
            # Update the document record
            cursor.execute('''
                UPDATE documents 
                SET filename = %s, file_path = %s, file_size = %s, file_type = %s
                WHERE id = %s
            ''', (file.filename, new_file_path, file_size, file_type, document_id))
            
            # Delete existing topic relationships
            cursor.execute('DELETE FROM document_topics WHERE document_id = %s', (document_id,))
            
            # Insert new topic relationships if topics are specified
            if topic_ids and len(topic_ids) > 0:
                for topic_id in topic_ids:
                    cursor.execute('''
                        INSERT INTO document_topics (document_id, topic_id)
                        VALUES (%s, %s)
                    ''', (document_id, topic_id))
        else:
            # If document doesn't exist in database, create it
            cursor.execute('''
                INSERT INTO documents (filename, file_path, file_size, file_type)
                VALUES (%s, %s, %s, %s) RETURNING id
            ''', (file.filename, new_file_path, file_size, file_type))
            
            result = cursor.fetchone()
            if result is None:
                raise Exception("Failed to insert document record")
            document_id = result[0]
            
            # Insert topic relationships if topics are specified
            if topic_ids and len(topic_ids) > 0:
                for topic_id in topic_ids:
                    cursor.execute('''
                        INSERT INTO document_topics (document_id, topic_id)
                        VALUES (%s, %s)
                    ''', (document_id, topic_id))
        
        conn.commit()
        conn.close()
        
        # Clean up vector database and rebuild with new document
        global ai_app, ai_app_ready
        try:
            if ai_app and ai_app.db:
                # Remove old document from vector database first
                if hasattr(ai_app, 'remove_document_from_vector_db'):
                    if ai_app.remove_document_from_vector_db(old_filename):
                        add_backend_log(f"Old document {old_filename} removed from vector database")
                    else:
                        add_backend_log(f"Failed to remove old document from vector database, rebuilding")
                        ai_app.build_db("replace_rebuild")
                else:
                    # No incremental method, rebuild
                    ai_app.build_db("replace_rebuild")
                
                # Add new document incrementally
                if ai_app.add_document_incremental(new_file_path):
                    add_backend_log(f"New document {file.filename} added to vector database")
                else:
                    add_backend_log(f"Failed to add new document incrementally, rebuilding database")
                    ai_app.build_db("replace_fallback_rebuild")
                
                ai_app_ready = True
            else:
                # No AI app, create new one
                ai_app = AIApp(None)
                ai_app.build_db("replace_initial_build")
                ai_app_ready = True
            
            add_backend_log(f"Vector database updated after replacing {old_filename} with {file.filename}")
        except Exception as e:
            add_backend_log(f"Error updating vector database: {e}")
            # Force rebuild as fallback
            try:
                ai_app = AIApp(None)
                ai_app.build_db("replace_error_rebuild")
                ai_app_ready = True
                add_backend_log(f"Vector database rebuilt after error")
            except Exception as rebuild_error:
                add_backend_log(f"Failed to rebuild vector database: {rebuild_error}")
                ai_app_ready = False
        
        add_backend_log(f"Document replaced: {old_filename} with {file.filename}")
        return jsonify({'status': 'success', 'message': f'Document {old_filename} replaced with {file.filename}'})
    except Exception as e:
        add_backend_log(f"Error message: {e}")
        return jsonify({'error': str(e)})

@app.route('/delete_document', methods=['POST'])
def delete_document():
    """Delete a document from the database and filesystem, then rebuild the vector DB."""
    try:
        data = request.get_json()
        filename = data.get('filename')
        if not filename:
            return jsonify({'error': 'Filename is required'})
        
        file_path = os.path.join('your_docs', filename)
        
        # Delete the file from the filesystem if it exists
        if os.path.exists(file_path):
            os.remove(file_path)
            add_backend_log(f"File deleted from filesystem: {filename}")
        else:
            add_backend_log(f"File not found on disk: {file_path}")
        
        # Delete from the database with proper cleanup
        conn = get_db()
        cursor = conn.cursor()
        
        # First, get the document ID for cleanup
        cursor.execute('SELECT id FROM documents WHERE filename = %s', (filename,))
        result = cursor.fetchone()
        
        if result:
            document_id = result[0]
            
            # Clean up document-topic relationships first (CASCADE should handle this, but being explicit)
            cursor.execute('DELETE FROM document_topics WHERE document_id = %s', (document_id,))
            add_backend_log(f"Cleaned up {cursor.rowcount} topic relationships for {filename}")
            
            # Delete the document record
            cursor.execute('DELETE FROM documents WHERE id = %s', (document_id,))
            add_backend_log(f"Document record deleted from database: {filename}")
            
            # Check for orphaned topics (topics with no documents)
            cursor.execute('''
                DELETE FROM topics t 
                WHERE NOT EXISTS (
                    SELECT 1 FROM document_topics dt WHERE dt.topic_id = t.id
                )
            ''')
            orphaned_topics_count = cursor.rowcount
            if orphaned_topics_count > 0:
                add_backend_log(f"Cleaned up {orphaned_topics_count} orphaned topics")
            
            conn.commit()
            add_backend_log(f"Database cleanup completed for {filename}")
        else:
            add_backend_log(f"Document {filename} not found in database")
        
        conn.close()
        
        # Clean up vector database and rebuild
        global ai_app, ai_app_ready
        try:
            if ai_app and ai_app.db:
                # Try incremental cleanup first
                if hasattr(ai_app, 'remove_document_from_vector_db'):
                    if ai_app.remove_document_from_vector_db(filename):
                        add_backend_log(f"Document {filename} removed from vector database incrementally")
                    else:
                        # Fallback to full rebuild
                        add_backend_log(f"Incremental vector cleanup failed, rebuilding database for {filename}")
                        ai_app.build_db("delete_rebuild")
                else:
                    # No incremental method, rebuild
                    ai_app.build_db("delete_rebuild")
                ai_app_ready = True
            else:
                # No AI app, create new one
                ai_app = AIApp(None)
                ai_app.build_db("delete_rebuild")
                ai_app_ready = True
            
            add_backend_log(f"Vector database updated after deleting {filename}")
        except Exception as e:
            add_backend_log(f"Error updating vector database: {e}")
            # Force rebuild as fallback
            try:
                ai_app = AIApp(None)
                ai_app.build_db("delete_rebuild_fallback")
                ai_app_ready = True
                add_backend_log(f"Vector database rebuilt after error")
            except Exception as rebuild_error:
                add_backend_log(f"Failed to rebuild vector database: {rebuild_error}")
                ai_app_ready = False
        
        add_backend_log(f"Document {filename} completely deleted and cleaned up")
        return jsonify({'status': 'success', 'message': f'Document {filename} deleted successfully'})
    except Exception as e:
        add_backend_log(f"Error deleting document {filename}: {e}")
        return jsonify({'error': str(e)})

@app.route('/bulk_delete_documents', methods=['POST'])
def bulk_delete_documents():
    """Delete multiple documents efficiently with proper error handling and real-time updates."""
    try:
        data = request.get_json()
        filenames = data.get('filenames', [])
        
        if not filenames:
            return jsonify({'error': 'No filenames provided'})
        
        add_backend_log(f"Starting bulk delete of {len(filenames)} documents")
        
        # Use a thread pool to process deletions concurrently
        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all delete tasks
            future_to_filename = {
                executor.submit(delete_single_document, filename): filename 
                for filename in filenames
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_filename):
                filename = future_to_filename[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        'filename': filename,
                        'status': 'error',
                        'error': str(e)
                    })
        
        # Count successes and failures
        successful_deletes = [r for r in results if r['status'] == 'success']
        failed_deletes = [r for r in results if r['status'] == 'error']
        
        # Rebuild the AI database once after all deletions
        if successful_deletes:
            try:
                global ai_app, ai_app_ready
                ai_app = AIApp(None)
                ai_app.build_db("bulk_delete_rebuild")
                ai_app_ready = True
                add_backend_log(f"Database rebuilt after bulk delete of {len(successful_deletes)} documents")
            except Exception as e:
                add_backend_log(f"Error rebuilding database after bulk delete: {e}")
        
        return jsonify({
            'status': 'success',
            'message': f'Bulk delete completed: {len(successful_deletes)} successful, {len(failed_deletes)} failed',
            'results': results,
            'successful_count': len(successful_deletes),
            'failed_count': len(failed_deletes)
        })
        
    except Exception as e:
        add_backend_log(f"Error in bulk delete: {e}")
        return jsonify({'error': str(e)})

def delete_single_document(filename):
    """Helper function to delete a single document with proper error handling."""
    try:
        file_path = os.path.join('your_docs', filename)
        
        # Delete from filesystem
        if os.path.exists(file_path):
            os.remove(file_path)
            add_backend_log(f"File deleted from filesystem: {filename}")
        
        # Delete from database with proper cleanup
        conn = get_db()
        cursor = conn.cursor()
        
        # First, get the document ID for cleanup
        cursor.execute('SELECT id FROM documents WHERE filename = %s', (filename,))
        result = cursor.fetchone()
        
        if result:
            document_id = result[0]
            
            # Clean up document-topic relationships first
            cursor.execute('DELETE FROM document_topics WHERE document_id = %s', (document_id,))
            topic_relationships_cleaned = cursor.rowcount
            add_backend_log(f"Cleaned up {topic_relationships_cleaned} topic relationships for {filename}")
            
            # Delete the document record
            cursor.execute('DELETE FROM documents WHERE id = %s', (document_id,))
            add_backend_log(f"Document record deleted from database: {filename}")
            
            conn.commit()
            add_backend_log(f"Database cleanup completed for {filename}")
        else:
            add_backend_log(f"Document {filename} not found in database")
        
        conn.close()
        
        add_backend_log(f"Successfully deleted: {filename}")
        return {
            'filename': filename,
            'status': 'success',
            'message': f'Document {filename} deleted successfully'
        }
        
    except Exception as e:
        add_backend_log(f"Error deleting {filename}: {e}")
        return {
            'filename': filename,
            'status': 'error',
            'error': str(e)
        }

@app.route('/bulk_upload_documents', methods=['POST'])
def bulk_upload_documents():
    """Upload multiple documents efficiently with proper error handling."""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files uploaded'})
        
        files = request.files.getlist('files')
        topic_ids_json = request.form.get('topic_ids', '[]')
        topic_ids = json.loads(topic_ids_json)
        topic_ids = [int(tid) for tid in topic_ids if str(tid).isdigit()]
        
        if not files:
            return jsonify({'error': 'No files selected'})
        
        add_backend_log(f"Starting bulk upload of {len(files)} documents")
        
        # Process uploads concurrently
        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all upload tasks
            future_to_file = {
                executor.submit(upload_single_document, file, topic_ids): file 
                for file in files
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        'filename': file.filename,
                        'status': 'error',
                        'error': str(e)
                    })
        
        # Count successes and failures
        successful_uploads = [r for r in results if r['status'] == 'success']
        failed_uploads = [r for r in results if r['status'] == 'error']
        conflicts = [r for r in results if r['status'] == 'conflict']
        
        # Rebuild the AI database once after all uploads
        if successful_uploads:
            try:
                global ai_app, ai_app_ready
                if ai_app and ai_app.db:
                    # Use incremental update for existing database
                    add_backend_log(f"Updating existing database with {len(successful_uploads)} documents")
                    for result in successful_uploads:
                        file_path = os.path.join('your_docs', result['filename'])
                        if not ai_app.add_document_incremental(file_path):
                            add_backend_log(f"Incremental update failed for {result['filename']}, falling back to rebuild")
                            ai_app.build_db("bulk_upload_rebuild")
                            break
                else:
                    # Do initial build for new database
                    add_backend_log(f"Building new database for {len(successful_uploads)} documents")
                    ai_app = AIApp(None)
                    ai_app.build_db("bulk_upload_build")
                
                ai_app_ready = True
                add_backend_log(f"Database updated after bulk upload of {len(successful_uploads)} documents")
            except Exception as e:
                add_backend_log(f"Error updating database after bulk upload: {e}")
                # Force rebuild as fallback
                try:
                    ai_app = AIApp(None)
                    ai_app.build_db("bulk_upload_fallback")
                    ai_app_ready = True
                    add_backend_log(f"Database rebuilt after bulk upload error")
                except Exception as rebuild_error:
                    add_backend_log(f"Failed to rebuild database after bulk upload: {rebuild_error}")
                    ai_app_ready = False
        
        return jsonify({
            'status': 'success',
            'message': f'Bulk upload completed: {len(successful_uploads)} successful, {len(failed_uploads)} failed, {len(conflicts)} conflicts',
            'results': results,
            'successful_count': len(successful_uploads),
            'failed_count': len(failed_uploads),
            'conflict_count': len(conflicts)
        })
        
    except Exception as e:
        add_backend_log(f"Error in bulk upload: {e}")
        return jsonify({'error': str(e)})

def upload_single_document(file, topic_ids):
    """Helper function to upload a single document with proper error handling."""
    try:
        if file.filename == '':
            return {
                'filename': 'unknown',
                'status': 'error',
                'error': 'No file selected'
            }
        
        # Validate file type
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            return {
                'filename': file.filename,
                'status': 'error',
                'error': f'File type {file_ext} not allowed'
            }
        
        # Check for filename conflict
        docs_dir = DOCS_PATH
        os.makedirs(docs_dir, exist_ok=True)
        file_path = os.path.join(docs_dir, file.filename)
        
        if os.path.exists(file_path):
            return {
                'filename': file.filename,
                'status': 'conflict',
                'message': f'File "{file.filename}" already exists'
            }
        
        # Save file
        file.save(file_path)
        
        # Save document info to database
        conn = get_db()
        cursor = conn.cursor()
        
        file_size = os.path.getsize(file_path)
        file_type = file_ext.upper()
        
        # Insert document record
        cursor.execute('''
            INSERT INTO documents (filename, file_path, file_size, file_type)
            VALUES (%s, %s, %s, %s) RETURNING id
        ''', (file.filename, file_path, file_size, file_type))
        
        result = cursor.fetchone()
        if result is None:
            raise Exception("Failed to insert document record")
        document_id = result[0]
        
        # Insert topic relationships if topics are specified
        if topic_ids and len(topic_ids) > 0:
            for topic_id in topic_ids:
                cursor.execute('''
                    INSERT INTO document_topics (document_id, topic_id)
                    VALUES (%s, %s)
                ''', (document_id, topic_id))
        
        conn.commit()
        conn.close()
        
        add_backend_log(f"Successfully uploaded: {file.filename}")
        return {
            'filename': file.filename,
            'status': 'success',
            'message': f'Document {file.filename} uploaded successfully'
        }
        
    except Exception as e:
        add_backend_log(f"Error uploading {file.filename}: {e}")
        return {
            'filename': file.filename,
            'status': 'error',
            'error': str(e)
        }

@app.route('/get_topics', methods=['GET'])
def get_topics():
    """Get list of all topics"""
    try:
        conn = get_db()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cursor.execute('''
            SELECT id, name, description, created_at 
            FROM topics 
            ORDER BY name
        ''')
        
        topics = cursor.fetchall()
        conn.close()
        
        return jsonify({'topics': topics})
    except Exception as e:
        add_backend_log(f"Error message: {e}")
        return jsonify({'error': str(e)})

@app.route('/create_topic', methods=['POST'])
def create_topic():
    """Create a new topic"""
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        description = data.get('description', '').strip()
        
        if not name:
            return jsonify({'error': 'Topic name is required'})
        
        conn = get_db()
        cursor = conn.cursor()
        
        # Check if topic already exists
        cursor.execute('SELECT id FROM topics WHERE name = %s', (name,))
        if cursor.fetchone():
            conn.close()
            return jsonify({'error': f'Topic "{name}" already exists'})
        
        # Create new topic
        cursor.execute('''
            INSERT INTO topics (name, description) 
            VALUES (%s, %s) 
            RETURNING id
        ''', (name, description))
        
        result = cursor.fetchone()
        if result is None:
            conn.close()
            return jsonify({'error': 'Failed to create topic'})
        
        topic_id = result[0]
        conn.commit()
        conn.close()
        
        add_backend_log(f"Topic created: {name}")
        return jsonify({
            'status': 'success', 
            'message': f'Topic "{name}" created successfully',
            'topic_id': topic_id
        })
    except Exception as e:
        add_backend_log(f"Error message: {e}")
        return jsonify({'error': str(e)})

@app.route('/delete_topics', methods=['POST'])
def delete_topics():
    """Delete multiple topics and remove their tags from documents"""
    try:
        data = request.get_json()
        topic_ids = data.get('topic_ids', [])
        if not topic_ids:
            return jsonify({'error': 'No topic IDs provided'})
        # Convert all topic_ids to integers (handle string IDs from frontend)
        topic_ids = [int(tid) for tid in topic_ids if str(tid).isdigit()]
        if not topic_ids:
            return jsonify({'error': 'No valid topic IDs provided'})
        conn = get_db()
        cursor = conn.cursor()
        # Get topic names for logging
        cursor.execute('SELECT name FROM topics WHERE id = ANY(%s)', (topic_ids,))
        topic_names = [row[0] for row in cursor.fetchall()]
        # Delete topic relationships from documents (this removes the tags)
        cursor.execute('DELETE FROM document_topics WHERE topic_id = ANY(%s)', (topic_ids,))
        # Delete the topics themselves
        cursor.execute('DELETE FROM topics WHERE id = ANY(%s) RETURNING id', (topic_ids,))
        deleted_rows = cursor.fetchall()
        deleted_count = len(deleted_rows)
        conn.commit()
        conn.close()
        # Always reinitialize AI with updated documents
        global ai_app
        ai_app = AIApp(None)
        add_backend_log(f"Topics deleted: {', '.join(topic_names)}")
        return jsonify({
            'status': 'success', 
            'message': f'Successfully deleted {deleted_count} topic(s): {", ".join(topic_names)}',
            'deleted_count': deleted_count
        })
    except Exception as e:
        add_backend_log(f"Error message: {e}")
        return jsonify({'error': str(e)})

@app.route('/get_backend_logs', methods=['GET'])
def get_backend_logs():
    """Return the last 100 lines of backend logs as JSON for the frontend logs tab."""
    return jsonify({'logs': backend_logs[-100:]})

@app.route('/rebuild_database', methods=['POST'])
def rebuild_database():
    """Force rebuild the vector database."""
    global ai_app
    if not ai_app:
        return jsonify({'error': 'AI Assistant not initialized'})
    
    try:
        add_backend_log("Starting forced database rebuild...")
        ai_app.force_rebuild_db()
        add_backend_log("Database rebuild completed successfully")
        return jsonify({'success': True, 'message': 'Database rebuilt successfully'})
    except Exception as e:
        add_backend_log(f"Error rebuilding database: {str(e)}")
        return jsonify({'error': f'Failed to rebuild database: {str(e)}'})

@app.route('/test_log', methods=['GET'])
def test_log():
    add_backend_log("TEST LOG: /test_log endpoint was called.")
    return jsonify({'status': 'success', 'message': 'Test log written.'})

@app.route('/test_db', methods=['GET'])
def test_db():
    """Test database connection and table status"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('documents', 'topics', 'document_topics')
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        # Count documents
        cursor.execute('SELECT COUNT(*) FROM documents')
        doc_count = cursor.fetchone()[0]
        
        # Count topics
        cursor.execute('SELECT COUNT(*) FROM topics')
        topic_count = cursor.fetchone()[0]
        
        # Count document-topic relationships
        cursor.execute('SELECT COUNT(*) FROM document_topics')
        relationship_count = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'status': 'success',
            'tables_exist': tables,
            'document_count': doc_count,
            'topic_count': topic_count,
            'relationship_count': relationship_count
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})

@app.route('/clear_backend_logs', methods=['POST'])
def clear_backend_logs():
    """Clear the in-memory backend logs when session ends"""
    backend_logs.clear()
    return jsonify({'status': 'success', 'message': 'Backend logs cleared.'})

@app.route('/cleanup_orphaned_data', methods=['POST'])
def cleanup_orphaned_data():
    """Clean up orphaned data in the database"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Clean up orphaned topics (topics with no documents)
        cursor.execute('''
            DELETE FROM topics t 
            WHERE NOT EXISTS (
                SELECT 1 FROM document_topics dt WHERE dt.topic_id = t.id
            )
        ''')
        orphaned_topics_count = cursor.rowcount
        
        # Clean up orphaned document-topic relationships (documents that no longer exist)
        cursor.execute('''
            DELETE FROM document_topics dt 
            WHERE NOT EXISTS (
                SELECT 1 FROM documents d WHERE d.id = dt.document_id
            )
        ''')
        orphaned_relationships_count = cursor.rowcount
        
        # Clean up orphaned document records (files that no longer exist on disk)
        orphaned_documents_count = 0
        cursor.execute('SELECT id, filename FROM documents')
        documents = cursor.fetchall()
        
        for doc_id, filename in documents:
            file_path = os.path.join('your_docs', filename)
            if not os.path.exists(file_path):
                cursor.execute('DELETE FROM documents WHERE id = %s', (doc_id,))
                orphaned_documents_count += 1
                add_backend_log(f"Removed orphaned document record: {filename}")
        
        conn.commit()
        conn.close()
        
        add_backend_log(f"Cleanup completed: {orphaned_topics_count} orphaned topics, {orphaned_relationships_count} orphaned relationships, {orphaned_documents_count} orphaned documents")
        
        return jsonify({
            'status': 'success',
            'message': 'Orphaned data cleanup completed',
            'orphaned_topics': orphaned_topics_count,
            'orphaned_relationships': orphaned_relationships_count,
            'orphaned_documents': orphaned_documents_count
        })
        
    except Exception as e:
        add_backend_log(f"Error during orphaned data cleanup: {e}")
        return jsonify({'error': str(e)})

@app.route('/cleanup_ghost_documents', methods=['POST'])
def cleanup_ghost_documents():
    """Comprehensive cleanup of ghost documents and rebuild vector database"""
    try:
        add_backend_log("Starting comprehensive ghost document cleanup...")
        
        # Step 1: Clean up orphaned data
        try:
            cleanup_result = cleanup_orphaned_data()
            if isinstance(cleanup_result, dict) and cleanup_result.get('status') == 'success':
                add_backend_log("Orphaned data cleanup completed successfully")
            else:
                add_backend_log("Orphaned data cleanup failed")
        except Exception as cleanup_error:
            add_backend_log(f"Error during orphaned data cleanup: {cleanup_error}")
            # Continue with vector database rebuild even if cleanup fails
        
        # Step 2: Force rebuild of vector database
        global ai_app, ai_app_ready
        try:
            add_backend_log("Rebuilding vector database after cleanup...")
            ai_app = AIApp(None)
            ai_app.build_db("ghost_cleanup_rebuild")
            ai_app_ready = True
            add_backend_log("Vector database rebuilt successfully after ghost cleanup")
        except Exception as e:
            add_backend_log(f"Error rebuilding vector database: {e}")
            ai_app_ready = False
            return jsonify({'error': f'Failed to rebuild vector database: {str(e)}'})
        
        # Step 3: Verify cleanup
        conn = get_db()
        cursor = conn.cursor()
        
        # Count remaining documents
        cursor.execute('SELECT COUNT(*) FROM documents')
        remaining_docs = cursor.fetchone()[0]
        
        # Count remaining topics
        cursor.execute('SELECT COUNT(*) FROM topics')
        remaining_topics = cursor.fetchone()[0]
        
        # Count remaining relationships
        cursor.execute('SELECT COUNT(*) FROM document_topics')
        remaining_relationships = cursor.fetchone()[0]
        
        conn.close()
        
        add_backend_log(f"Ghost cleanup completed. Remaining: {remaining_docs} docs, {remaining_topics} topics, {remaining_relationships} relationships")
        
        return jsonify({
            'status': 'success',
            'message': 'Ghost document cleanup completed successfully',
            'remaining_documents': remaining_docs,
            'remaining_topics': remaining_topics,
            'remaining_relationships': remaining_relationships
        })
        
    except Exception as e:
        add_backend_log(f"Error during ghost document cleanup: {e}")
        return jsonify({'error': str(e)})

@app.route('/test_has_documents', methods=['GET'])
def test_has_documents():
    """Test endpoint to check if documents exist"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Check total documents
        cursor.execute('SELECT COUNT(*) FROM documents')
        total_docs = cursor.fetchone()[0]
        
        # Check topics
        cursor.execute('SELECT COUNT(*) FROM topics')
        total_topics = cursor.fetchone()[0]
        
        # Check document-topic relationships
        cursor.execute('SELECT COUNT(*) FROM document_topics')
        total_relationships = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'status': 'success',
            'total_documents': total_docs,
            'total_topics': total_topics,
            'total_relationships': total_relationships
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})

@app.route('/has_documents_for_topics', methods=['POST'])
def has_documents_for_topics():
    """Check if documents exist for the given topic IDs"""
    try:
        data = request.get_json()
        topic_ids = data.get('topic_ids', [])
        
        add_backend_log(f"Checking documents for topics: {topic_ids}")
        
        conn = get_db()
        cursor = conn.cursor()
        
        if not topic_ids:
            # Check if any documents exist at all
            cursor.execute('SELECT COUNT(*) FROM documents')
            count = cursor.fetchone()[0]
            conn.close()
            add_backend_log(f"Found {count} total documents")
            return jsonify({'has_documents': count > 0})
        else:
            # Check if documents exist for specific topics
            cursor.execute('''
                SELECT COUNT(DISTINCT d.id) 
                FROM documents d 
                JOIN document_topics dt ON d.id = dt.document_id 
                WHERE dt.topic_id = ANY(%s)
            ''', (topic_ids,))
            count = cursor.fetchone()[0]
            conn.close()
            add_backend_log(f"Found {count} documents for topics {topic_ids}")
            return jsonify({'has_documents': count > 0})
            
    except Exception as e:
        add_backend_log(f"Error checking documents for topics: {e}")
        return jsonify({'has_documents': False, 'error': str(e)})

@app.route('/validate_vector_db', methods=['POST'])
def validate_vector_db():
    """Validate vector database integrity and clean any stale data."""
    try:
        global ai_app, ai_app_ready
        
        if not ai_app:
            ai_app = AIApp(None)
        
        add_backend_log("Starting vector database integrity validation...")
        
        # Validate the vector database
        validation_result = ai_app.validate_vector_db_integrity()
        
        if validation_result.get('status') == 'rebuilt':
            add_backend_log(f"Vector database rebuilt: {validation_result.get('message')}")
            ai_app_ready = True
        elif validation_result.get('status') == 'valid':
            add_backend_log(f"Vector database integrity confirmed: {validation_result.get('message')}")
        else:
            add_backend_log(f"Vector database validation issue: {validation_result.get('message')}")
        
        return jsonify({
            'status': 'success',
            'validation_result': validation_result
        })
        
    except Exception as e:
        add_backend_log(f"Error validating vector database: {e}")
        return jsonify({'error': str(e)})

@app.route('/vector_db_health', methods=['GET'])
def vector_db_health():
    """Get comprehensive health status of the vector database."""
    try:
        global ai_app, ai_app_ready
        
        if not ai_app:
            ai_app = AIApp(None)
        
        health_status = {
            'ai_app_ready': ai_app_ready,
            'vector_db_exists': ai_app.db is not None,
            'disk_files': [],
            'db_sources': [],
            'integrity_status': 'unknown',
            'total_chunks': 0,
            'memory_usage': {},
            'last_operation': 'none'
        }
        
        # Get disk files
        try:
            for root, _, files in os.walk('your_docs'):
                for file in files:
                    health_status['disk_files'].append(file)
        except Exception as e:
            health_status['disk_files_error'] = str(e)
        
        # Get vector database sources if it exists
        if ai_app.db:
            try:
                # Get a sample of documents to analyze
                sample_docs = ai_app.db.similarity_search("", k=1000)
                health_status['total_chunks'] = len(sample_docs)
                
                # Extract unique sources
                sources = set()
                for doc in sample_docs:
                    source = doc.metadata.get('source', '')
                    if source:
                        sources.add(os.path.basename(source))
                health_status['db_sources'] = list(sources)
                
                # Check for discrepancies
                disk_files_set = set(health_status['disk_files'])
                db_sources_set = set(health_status['db_sources'])
                
                stale_sources = db_sources_set - disk_files_set
                missing_sources = disk_files_set - db_sources_set
                
                if stale_sources or missing_sources:
                    health_status['integrity_status'] = 'compromised'
                    health_status['stale_sources'] = list(stale_sources)
                    health_status['missing_sources'] = list(missing_sources)
                else:
                    health_status['integrity_status'] = 'healthy'
                    
            except Exception as e:
                health_status['db_analysis_error'] = str(e)
                health_status['integrity_status'] = 'error'
        else:
            health_status['integrity_status'] = 'no_database'
        
        # Get memory usage information
        try:
            import psutil
            memory = psutil.virtual_memory()
            health_status['memory_usage'] = {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'used_gb': round(memory.used / (1024**3), 2),
                'percent': memory.percent
            }
        except Exception as e:
            health_status['memory_error'] = str(e)
        
        # Get last operation from in-memory logs
        try:
            if backend_logs:
                last_log = backend_logs[-1]
                health_status['last_operation'] = f"{last_log}"
            else:
                health_status['last_operation'] = "No operations logged yet"
        except Exception as e:
            health_status['log_error'] = str(e)
        
        return jsonify({
            'status': 'success',
            'health': health_status
        })
        
    except Exception as e:
        add_backend_log(f"Error getting vector database health: {e}")
        return jsonify({'error': str(e)})

@app.route('/questionnaire_progress/<job_id>', methods=['GET'])
def get_questionnaire_progress(job_id):
    """Get progress of questionnaire processing"""
    if job_id in questionnaire_progress:
        return jsonify(questionnaire_progress[job_id])
    else:
        return jsonify({'error': 'Job not found'})

@app.route('/force_cleanup_and_rebuild', methods=['POST'])
def force_cleanup_and_rebuild():
    """Force a complete cleanup of both PostgreSQL and vector databases, then rebuild everything."""
    try:
        add_backend_log("Starting forced cleanup and rebuild of all databases...")
        
        # Step 1: Clean up PostgreSQL database completely
        conn = get_db()
        cursor = conn.cursor()
        
        # Get all document filenames from PostgreSQL
        cursor.execute('SELECT filename FROM documents')
        postgres_files = [row[0] for row in cursor.fetchall()]
        add_backend_log(f"Found {len(postgres_files)} documents in PostgreSQL: {postgres_files}")
        
        # Check which files actually exist on disk
        existing_files = []
        missing_files = []
        for filename in postgres_files:
            file_path = os.path.join('your_docs', filename)
            if os.path.exists(file_path):
                existing_files.append(filename)
            else:
                missing_files.append(filename)
        
        add_backend_log(f"Files on disk: {existing_files}")
        add_backend_log(f"Missing files (will be cleaned from PostgreSQL): {missing_files}")
        
        # Clean up missing files from PostgreSQL
        if missing_files:
            for filename in missing_files:
                cursor.execute('SELECT id FROM documents WHERE filename = %s', (filename,))
                result = cursor.fetchone()
                if result:
                    document_id = result[0]
                    # Clean up document-topic relationships
                    cursor.execute('DELETE FROM document_topics WHERE document_id = %s', (document_id,))
                    # Delete the document record
                    cursor.execute('DELETE FROM documents WHERE id = %s', (document_id,))
                    add_backend_log(f"Cleaned up missing document from PostgreSQL: {filename}")
        
        # Check for orphaned topics
        cursor.execute('''
            DELETE FROM topics t 
            WHERE NOT EXISTS (
                SELECT 1 FROM document_topics dt WHERE dt.topic_id = t.id
            )
        ''')
        orphaned_topics_count = cursor.rowcount
        if orphaned_topics_count > 0:
            add_backend_log(f"Cleaned up {orphaned_topics_count} orphaned topics")
        
        conn.commit()
        conn.close()
        add_backend_log("PostgreSQL cleanup completed")
        
        # Step 2: Force rebuild of vector database
        global ai_app, ai_app_ready
        try:
            if ai_app:
                ai_app.force_rebuild_db()
            else:
                ai_app = AIApp(None)
                ai_app.force_rebuild_db()
            ai_app_ready = True
            add_backend_log("Vector database force rebuild completed")
        except Exception as e:
            add_backend_log(f"Error in vector database rebuild: {e}")
            return jsonify({'error': f'Vector database rebuild failed: {str(e)}'})
        
        # Step 3: Verify synchronization
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM documents')
        postgres_count = cursor.fetchone()[0]
        conn.close()
        
        # Get vector DB count
        vector_count = 0
        if ai_app and ai_app.db and hasattr(ai_app.db, 'index'):
            vector_count = ai_app.db.index.ntotal
        
        add_backend_log(f"Database synchronization complete - PostgreSQL: {postgres_count}, Vector DB: {vector_count}")
        
        return jsonify({
            'status': 'success', 
            'message': 'Complete cleanup and rebuild successful',
            'postgres_count': postgres_count,
            'vector_count': vector_count
        })
        
    except Exception as e:
        add_backend_log(f"Error in force cleanup and rebuild: {e}")
        return jsonify({'error': str(e)})

@app.route('/check_database_integrity', methods=['POST'])
def check_database_integrity():
    """Manually check database integrity without forcing a rebuild."""
    try:
        global ai_app
        if not ai_app:
            return jsonify({'error': 'AI application not initialized'}), 500
        
        add_backend_log("Starting manual database integrity check...")
        
        # Run the smart integrity check
        result = ai_app.periodic_integrity_check()
        
        status = result.get('status', 'unknown')
        message = result.get('message', 'Unknown status')
        
        if status == 'valid':
            add_backend_log(f"✅ Database integrity check: {message}")
        elif status == 'minor_issues':
            add_backend_log(f"⚠️ Database integrity check: {message}")
        elif status == 'moderate_issues':
            add_backend_log(f"⚠️ Database integrity check: {message}")
        elif status == 'fixed':
            add_backend_log(f"🔧 Database integrity check: {message}")
        elif status == 'rebuilt':
            add_backend_log(f"🔄 Database integrity check: {message}")
        elif status == 'skipped':
            add_backend_log(f"⏭️ Database integrity check: {message}")
        else:
            add_backend_log(f"❓ Database integrity check: {message}")
        
        return jsonify({
            'success': True,
            'status': status,
            'message': message,
            'details': result
        })
        
    except Exception as e:
        add_backend_log(f"Error in database integrity check: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/check_database_status', methods=['GET'])
def check_database_status():
    """Check the current status of both PostgreSQL and vector databases."""
    try:
        status = {}
        
        # Check PostgreSQL database
        conn = get_db()
        cursor = conn.cursor()
        
        # Get document count and filenames
        cursor.execute('SELECT COUNT(*) FROM documents')
        postgres_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT filename FROM documents ORDER BY filename')
        postgres_files = [row[0] for row in cursor.fetchall()]
        
        # Get topic count
        cursor.execute('SELECT COUNT(*) FROM topics')
        topic_count = cursor.fetchone()[0]
        
        conn.close()
        
        status['postgresql'] = {
            'document_count': postgres_count,
            'topic_count': topic_count,
            'documents': postgres_files
        }
        
        # Check vector database
        global ai_app
        if ai_app and ai_app.db and hasattr(ai_app.db, 'index'):
            vector_count = ai_app.db.index.ntotal
            status['vector_db'] = {
                'chunk_count': vector_count,
                'index_type': 'HNSW' if hasattr(ai_app.db.index, 'hnsw') else 'Standard',
                'ready': True
            }
        else:
            status['vector_db'] = {
                'chunk_count': 0,
                'index_type': 'None',
                'ready': False
            }
        
        # Check filesystem
        docs_path = 'your_docs'
        if os.path.exists(docs_path):
            files_on_disk = [f for f in os.listdir(docs_path) if os.path.isfile(os.path.join(docs_path, f))]
            status['filesystem'] = {
                'document_count': len(files_on_disk),
                'documents': sorted(files_on_disk)
            }
        else:
            status['filesystem'] = {
                'document_count': 0,
                'documents': []
            }
        
        # Check for mismatches
        postgres_set = set(postgres_files)
        filesystem_set = set(status['filesystem']['documents'])
        
        status['synchronization'] = {
            'postgres_only': list(postgres_set - filesystem_set),
            'filesystem_only': list(filesystem_set - postgres_set),
            'both': list(postgres_set & filesystem_set),
            'synchronized': postgres_set == filesystem_set
        }
        
        return jsonify(status)
        
    except Exception as e:
        add_backend_log(f"Error checking database status: {e}")
        return jsonify({'error': str(e)})

# Guest session cleanup route removed - no longer needed with shared user system

# Guest session cleanup route removed - no longer needed with shared user system

# Clean up old progress entries every hour
def schedule_cleanup_tasks():
    """Schedule periodic cleanup tasks"""
    def cleanup_task():
        while True:
            try:
                time.sleep(3600)  # Run every hour
                cleanup_old_progress()
            except Exception as e:
                print(f"Error in cleanup task: {e}")
    
    cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
    cleanup_thread.start()
    print("Cleanup tasks scheduled to run every hour")

# Start cleanup tasks
schedule_cleanup_tasks()

# configure_session_for_user function removed - no longer needed with shared user system

if __name__ == '__main__':
    print('Flask app starting...')
    
    # Start background progress cleanup
    start_progress_cleanup()
    
    # Initialize AI App before starting Flask
    print('Initializing AI App...')
    try:
        ai_app = get_or_init_ai_app()
        if ai_app:
            print('✓ AI App initialized successfully')
        else:
            print('⚠ AI App initialization failed, will retry on first request')
    except Exception as e:
        print(f'⚠ AI App initialization error: {e}, will retry on first request')
    
    # Start the Flask app
    # Get port from environment variable (Cloud Run) or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)