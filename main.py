from flask import Flask, render_template, request, jsonify, session, send_file
import os
from ai_bot import AIApp
from Logistics_Files.config_details import DOCS_PATH, DB_PATH, EMBEDDING_MODEL, UPLOAD_PIN, MODEL_NAME, PASSWORD
from Logistics_Files.backend_log import add_backend_log, backend_logs
import io
import csv
from PyPDF2 import PdfReader
from docx import Document as DocxReader
import xml.etree.ElementTree as ET
import psycopg2
import psycopg2.extras
import hashlib
import secrets
import re
import json
from datetime import datetime, timedelta
import logging
from werkzeug.utils import secure_filename
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Centralized allowed file extensions
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.csv', '.xml', '.xlsx'}

# Test log entry on startup
add_backend_log("TEST LOG: Flask app started and logging is configured.")

app = Flask(__name__)
# Set secret key from environment variable or generate a secure random one
app.secret_key = os.environ.get('SECRET_KEY') or secrets.token_urlsafe(32)  # Needed for session support

# AI App with lazy initialization and background building
ai_app = None
ai_app_initializing = False
ai_app_ready = False
ai_app_error = None

# Session security settings
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)

# Database setup - PostgreSQL
DATABASE_URL = f'postgresql://postgres:{PASSWORD}@127.0.0.1:5432/chat_history'
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable must be set")

# Use environment variable for upload PIN
UPLOAD_PIN = os.getenv('UPLOAD_PIN', '1964')

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number"
    return True, "Password is strong"

def validate_username(username):
    """Validate username format"""
    if len(username) < 3:
        return False, "Username must be at least 3 characters long"
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        return False, "Username can only contain letters, numbers, and underscores"
    return True, "Username is valid"

def get_db():
    """Get database connection"""
    return psycopg2.connect(DATABASE_URL)

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
            email_verified BOOLEAN DEFAULT FALSE,
            verification_token VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create chat_history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,
            user_id INTEGER,
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

def hash_password(password):
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def get_user_id():
    """Get current user ID from session"""
    if 'user_id' in session:
        return session['user_id']
    return None

def get_session_id():
    """Get or create session ID"""
    if 'session_id' not in session:
        session['session_id'] = secrets.token_urlsafe(32)
    return session['session_id']

def get_chat_history(limit=None, session_key=None):
    """Get chat history from database for a given session_key"""
    conn = get_db()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    user_id = get_user_id()
    order = 'ASC'
    limit_clause = ''
    if limit is not None:
        limit_clause = f'LIMIT {int(limit)}'
    if session_key:
        cursor.execute(f'''
            SELECT question, answer, confidence, sources FROM chat_history 
            WHERE session_id = %s 
            ORDER BY timestamp {order} 
            {limit_clause}
        ''', (session_key,))
    elif user_id:
        cursor.execute(f'''
            SELECT question, answer, confidence, sources FROM chat_history 
            WHERE user_id = %s 
            ORDER BY timestamp {order} 
            {limit_clause}
        ''', (user_id,))
    else:
        session_id = get_session_id()
        cursor.execute(f'''
            SELECT question, answer, confidence, sources FROM chat_history 
            WHERE session_id = %s 
            ORDER BY timestamp {order} 
            {limit_clause}
        ''', (session_id,))
    history = cursor.fetchall()
    conn.close()
    return [{
        'question': row['question'], 
        'answer': row['answer'],
        'confidence': row['confidence'],
        'sources': row['sources']
    } for row in history]

def save_chat_history(question, answer, confidence, sources, session_key=None):
    """Save chat history to database with session_key"""
    conn = get_db()
    cursor = conn.cursor()
    user_id = get_user_id()
    session_id = session_key if session_key else get_session_id()
    try:
        confidence = float(confidence) if confidence is not None else None
    except Exception:
        confidence = None
    
    # Handle structured source data
    if isinstance(sources, dict) and 'summary' in sources:
        # Convert structured sources to string format for database storage
        sources_str = sources.get('summary', 'N/A')
    else:
        # Legacy string format
        sources_str = str(sources) if sources else 'N/A'
    
    cursor.execute('''
        INSERT INTO chat_history (user_id, session_id, question, answer, confidence, sources)
        VALUES (%s, %s, %s, %s, %s, %s)
    ''', (user_id, session_id, question, answer, confidence, sources_str))
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

def get_or_init_ai_app():
    """Get AI app instance with lazy initialization"""
    global ai_app, ai_app_initializing, ai_app_ready, ai_app_error
    
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
            add_backend_log("AI app created successfully for empty documents")
        except Exception as e:
            add_backend_log(f"Error creating AI app: {e}")
            ai_app_ready = True
            ai_app_error = str(e)
        return ai_app
    
    # Start initialization in background
    ai_app_initializing = True
    add_backend_log("Starting AI app initialization in background...")
    
    def init_ai_background():
        global ai_app, ai_app_initializing, ai_app_ready, ai_app_error
        try:
            add_backend_log("Creating AI app instance...")
            ai_app = AIApp(None)
            add_backend_log("AI app instance created successfully")
            
            # Check if database exists and is valid
            add_backend_log("Checking if database exists and is valid...")
            try:
                db_valid = ai_app.database_exists_and_valid()
                add_backend_log(f"Database validation result: {db_valid}")
                
                if db_valid:
                    add_backend_log("Loading existing vector database...")
                    ai_app.load_vector_db()
                    ai_app_ready = True
                    add_backend_log("AI app ready with existing database")
                else:
                    # Check if there are documents to process
                    add_backend_log("Checking for documents to process...")
                    if os.path.exists(DOCS_PATH) and any(os.listdir(DOCS_PATH)):
                        add_backend_log("Building vector database from documents...")
                        ai_app.build_db("initial_build")
                        ai_app_ready = True
                        add_backend_log("AI app ready with new database")
                    else:
                        add_backend_log("No documents found - AI app ready for uploads")
                        ai_app_ready = True
            except Exception as db_error:
                add_backend_log(f"Error during database validation/loading: {db_error}")
                # Set as ready anyway so the app can function
                ai_app_ready = True
                add_backend_log("AI app ready despite database error")
                
        except Exception as e:
            ai_app_error = str(e)
            add_backend_log(f"Error initializing AI app: {e}")
            # Set as ready anyway so the app can function
            ai_app_ready = True
            add_backend_log("AI app ready despite initialization error")
        finally:
            ai_app_initializing = False
            add_backend_log("Background initialization completed")
    
    # Start background thread with timeout
    thread = threading.Thread(target=init_ai_background, daemon=True)
    thread.start()
    
    # Set a timeout to prevent hanging
    def timeout_handler():
        global ai_app_initializing, ai_app_ready, ai_app_error
        time.sleep(30)  # 30 second timeout (increased to allow for database building)
        if ai_app_initializing:
            add_backend_log("Background initialization timed out - setting as ready")
            ai_app_initializing = False
            ai_app_ready = True
            ai_app_error = "Initialization timed out"
    
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
    
    return ai_app_ready


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    if not username or not email or not password:
        return jsonify({'error': 'All fields are required'})
    
    # Validate email format
    if not validate_email(email):
        return jsonify({'error': 'Please enter a valid email address'})
    
    # Validate password strength
    password_valid, password_message = validate_password(password)
    if not password_valid:
        return jsonify({'error': password_message})
    
    # Validate username
    username_valid, username_message = validate_username(username)
    if not username_valid:
        return jsonify({'error': username_message})
    
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        password_hash = hash_password(password)
        # verification_token = secrets.token_urlsafe(32)  # Not used
        
        cursor.execute('''
            INSERT INTO users (username, email, password_hash) 
            VALUES (%s, %s, %s) RETURNING id
        ''', (username, email, password_hash))
        result = cursor.fetchone()
        if result is None:
            conn.close()
            return jsonify({'error': 'Registration failed'})
        user_id = result[0]
        conn.commit()
        
        # Auto-login after registration
        session['user_id'] = user_id
        session['username'] = username
        
        conn.close()
        add_backend_log(f"User registered: {username} ({email})")
        return jsonify({
            'status': 'success', 
            'message': 'Registration successful!'
        })
    except psycopg2.IntegrityError as e:
        conn.close()
        if 'username' in str(e):
            return jsonify({'error': 'Username already exists'})
        elif 'email' in str(e):
            return jsonify({'error': 'Email already exists'})
        else:
            return jsonify({'error': 'Registration failed'})
    except Exception as e:
        conn.close()
        add_backend_log(f"Registration error: {e}")
        return jsonify({'error': 'An unexpected error occurred. Please try again later.'})

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password are required'})
    
    conn = get_db()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    cursor.execute('''
        SELECT id, username, password_hash, email_verified 
        FROM users WHERE username = %s
    ''', (username,))
    user = cursor.fetchone()
    conn.close()
    
    if user and user['password_hash'] == hash_password(password):
        if not user['email_verified']:
            return jsonify({'error': 'Please verify your email before logging in'})
        
        session['user_id'] = user['id']
        session['username'] = user['username']
        add_backend_log(f"User logged in: {username}")
        return jsonify({'status': 'success', 'message': 'Login successful'})
    else:
        return jsonify({'error': 'Invalid username or password'})

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return jsonify({'status': 'success', 'message': 'Logout successful'})

@app.route('/user_status', methods=['GET'])
def user_status():
    user_id = get_user_id()
    username = session.get('username')
    return jsonify({
        'logged_in': user_id is not None,
        'username': username,
        'user_id': user_id
    })

@app.route('/clear_history', methods=['POST'])
def clear_history():
    conn = get_db()
    cursor = conn.cursor()
    
    user_id = get_user_id()
    session_id = get_session_id()
    
    if user_id:
        cursor.execute('DELETE FROM chat_history WHERE user_id = %s', (user_id,))
    else:
        cursor.execute('DELETE FROM chat_history WHERE session_id = %s', (session_id,))
    
    conn.commit()
    conn.close()
    return jsonify({'status': 'success', 'message': 'Chat history cleared'})

@app.route('/export_history', methods=['POST'])
def export_history():
    conn = get_db()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    user_id = get_user_id()
    session_id = get_session_id()
    
    if user_id:
        cursor.execute('''
            SELECT question, answer, timestamp FROM chat_history 
            WHERE user_id = %s ORDER BY timestamp
        ''', (user_id,))
    else:
        cursor.execute('''
            SELECT question, answer, timestamp FROM chat_history 
            WHERE session_id = %s ORDER BY timestamp
        ''', (session_id,))
    
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
    
    pin = request.form.get('pin')
    if pin != UPLOAD_PIN:
        return jsonify({'error': 'Invalid security PIN.'})
    
    files = request.files.getlist('files')
    temp_paths = []
    for file in files:
        if file.filename:
            # Sanitize filename
            safe_filename = secure_filename(file.filename)
            temp_path = os.path.join('your_docs', safe_filename)
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
    
    try:
        result = ai_app.get_document_status()
        result['ready'] = True
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'status': 'ready_with_warning',
            'message': f'AI Assistant ready (with warning: {str(e)})',
            'ready': True,
            'warning': str(e)
        })

@app.route('/ask', methods=['POST'])
def ask_question():
    global ai_app, ai_app_ready, ai_app_initializing, ai_app_error
    
    if ai_app_error:
        return jsonify({'error': f'AI Assistant error: {ai_app_error}'})
    
    if not ai_app_ready:
        if ai_app_initializing:
            return jsonify({'error': 'AI Assistant is still initializing. Please wait a moment and try again.'})
        else:
            # Trigger initialization
            ai_app = get_or_init_ai_app()
            return jsonify({'error': 'AI Assistant is starting up. Please wait a moment and try again.'})
    
    if not ai_app:
        return jsonify({'error': 'AI Assistant not available'})
    
    data = request.get_json()
    question = data.get('question', '')
    topic_ids = data.get('topic_ids', ['none'])
    session_key = data.get('session_key', None)
    
    if not question:
        return jsonify({'error': 'Question is required'})
    
    chat_history = get_chat_history(session_key=session_key)
    
    # Handle topic selection logic
    if 'all' in topic_ids:
        topic_ids = ['all']
    elif 'none' in topic_ids or not topic_ids:
        topic_ids = []
    # Otherwise, topic_ids is a list of selected topic IDs
    
    result = ai_app.handle_question(question, chat_history=chat_history, topic_ids=topic_ids)
    if 'error' in result:
        return jsonify({'error': result['error']})
    
    save_chat_history(question, result['answer'], result['confidence'], result['sources'], session_key=session_key)
    return jsonify({
        'answer': result['answer'],
        'confidence': result['confidence'],
        'sources': result['sources']
    })

@app.route('/upload_questions', methods=['POST'])
def upload_questions():
    global ai_app
    if not ai_app:
        return jsonify({'error': 'AI Assistant not initialized'})
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    filename = file.filename
    if not filename:
        return jsonify({'error': 'No file selected'})
    ext = filename.split('.')[-1].lower()
    questions = []
    # Extract questions based on file type
    if ext == 'csv':
        stream = io.StringIO(file.stream.read().decode('utf-8'))
        reader = csv.reader(stream)
        for row in reader:
            if row:
                questions.append(row[0])
    elif ext == 'txt':
        content = file.stream.read().decode('utf-8')
        # Split on double newlines or numbered patterns
        question_patterns = [
            r'\n\s*\d+\)',  # 1), 2), etc.
            r'\n\s*Q\d+\.',  # Q1., Q2., etc.
            r'\n\s*Question\s+\d+:',  # Question 1:, Question 2:, etc.
            r'\n\s*\d+\.',  # 1., 2., etc.
        ]
        
        questions = []
        # Try to split on question patterns first
        for pattern in question_patterns:
            parts = re.split(pattern, content)
            if len(parts) > 1:
                # Skip the first part (before first question) and clean up each question
                for part in parts[1:]:
                    question = part.strip()
                    if question and len(question) > 10:  # Only keep substantial questions
                        questions.append(question)
                break
        
        # If no pattern worked, split on double newlines
        if not questions:
            parts = re.split(r'\n\s*\n', content)
            for part in parts:
                question = part.strip()
                if question and len(question) > 10:
                    questions.append(question)
    elif ext == 'pdf':
        file.stream.seek(0)
        pdf = PdfReader(io.BytesIO(file.stream.read()))
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        
        add_backend_log(f"PDF text length: {len(full_text)} characters")
        add_backend_log(f"PDF text preview: {full_text[:500]}...")
        
        # Split into questions more intelligently
        
        # More comprehensive question patterns
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
        if full_text.strip():
            # Try each pattern and see which one gives the most questions
            best_pattern = None
            best_questions = []
            
            for pattern in question_patterns:
                parts = re.split(pattern, full_text)
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
                add_backend_log("No questions found with numbered patterns, trying alternative methods...")
                
                # Try splitting on question marks
                question_mark_parts = re.split(r'\?', full_text)
                for part in question_mark_parts[:-1]:  # Skip the last part (after last question mark)
                    question = part.strip()
                    if question and len(question) > 10:
                        # Find the last sentence that ends with a question mark
                        sentences = re.split(r'[.!?]', question)
                        if sentences:
                            last_sentence = sentences[-1].strip()
                            if last_sentence and len(last_sentence) > 10:
                                questions.append(last_sentence + "?")
                
                if not questions:
                    # Split on double newlines as last resort
                    parts = re.split(r'\n\s*\n', full_text)
                    for part in parts:
                        question = part.strip()
                        if question and len(question) > 10:
                            questions.append(question)
                
                add_backend_log(f"Found {len(questions)} questions using alternative methods")
    elif ext == 'docx':
        file.stream.seek(0)
        doc = DocxReader(io.BytesIO(file.stream.read()))
        for para in doc.paragraphs:
            line = para.text.strip()
            if line:
                questions.append(line)
    elif ext == 'xml':
        file.stream.seek(0)
        tree = ET.parse(io.BytesIO(file.stream.read()))
        root = tree.getroot()
        text = ET.tostring(root, encoding='unicode', method='text')
        if text:
            questions = [line.strip() for line in text.splitlines() if line.strip()]
        else:
            questions = []
    elif ext == 'xlsx':
        file.stream.seek(0)
        try:
            import openpyxl
            workbook = openpyxl.load_workbook(io.BytesIO(file.stream.read()), read_only=True)
            # Get the first worksheet
            worksheet = workbook.active
            # Read all cells from the first column (A)
            for row in worksheet.iter_rows(min_row=1, max_col=1, values_only=True):
                if row[0] and str(row[0]).strip():  # Check if cell is not empty
                    question = str(row[0]).strip()
                    # Only add substantial questions (not just numbers or single words)
                    if len(question) > 10 and not question.isdigit():
                        questions.append(question)
            workbook.close()
        except Exception as e:
            return jsonify({'error': f'Error reading XLSX file: {str(e)}'})
    else:
        return jsonify({'error': 'Unsupported file type'})
    
    add_backend_log(f"Processing {len(questions)} questions from {filename}")
    
    # Debug: Log all raw questions found
    for i, q in enumerate(questions):
        add_backend_log(f"Raw question {i+1}: {q[:100]}...")
    
    # Clean up questions: remove question numbers and normalize to single lines
    cleaned_questions = []
    for i, q in enumerate(questions):
        # Remove question numbers at the beginning (1), 2), Q1., etc.)
        cleaned_q = re.sub(r'^\s*(?:\d+\)|Q\d+\.|Question\s+\d+:|\d+\.)\s*', '', q.strip())
        
        # Normalize to single line by replacing newlines with spaces
        cleaned_q = re.sub(r'\s*\n\s*', ' ', cleaned_q)
        
        # Remove extra whitespace and normalize spaces
        cleaned_q = re.sub(r'\s+', ' ', cleaned_q).strip()
        
        # Remove any remaining question marks at the beginning
        cleaned_q = re.sub(r'^\s*\?\s*', '', cleaned_q)
        
        # Ensure the question ends with proper punctuation
        if cleaned_q and not cleaned_q.endswith(('.', '?', '!')):
            cleaned_q += '?'
        
        if cleaned_q and len(cleaned_q) > 10:
            cleaned_questions.append(cleaned_q)
            add_backend_log(f"Cleaned question {len(cleaned_questions)}: {cleaned_q[:100]}...")
    
    questions = cleaned_questions
    
    # Debug: Log the first few cleaned questions
    for i, q in enumerate(questions[:3]):
        add_backend_log(f"Sample question {i+1}: {q}")
    
    # Get answers for each question
    results = []
    for i, q in enumerate(questions):
        if not q:
            continue
        try:
            add_backend_log(f"Processing question {i+1}/{len(questions)}: {q}")
            
            # Ensure the question is not empty and has reasonable length
            if not q or len(q.strip()) < 3:
                add_backend_log(f"Skipping question {i+1}: too short or empty")
                continue
                
            # Check if AI app is properly initialized
            if not ai_app or not hasattr(ai_app, 'db') or not ai_app.db:
                add_backend_log(f"AI app not properly initialized for question {i+1}")
                result = {'error': 'AI Assistant not properly initialized. Please upload documents first.'}
            else:
                result = ai_app.handle_question(q)
            
            if 'error' in result:
                add_backend_log(f"Error processing question {i+1}: {result['error']}")
                answer = f"Error: {result['error']}"
                confidence = 0
                sources_data = {'summary': 'Error occurred', 'detailed': 'Error occurred'}
                sources_for_db = 'Error occurred'
            else:
                answer = result.get('answer', 'No answer generated')
                confidence = result.get('confidence', 0)
                sources = result.get('sources', {})
                if isinstance(sources, dict):
                    # Keep the structured format for frontend display
                    sources_data = sources
                    # For database storage, use summary
                    sources_for_db = sources.get('summary', 'N/A')
                else:
                    sources_data = {'summary': str(sources), 'detailed': str(sources)}
                    sources_for_db = str(sources)
                
            results.append({
                'question': q, 
                'answer': answer, 
                'confidence': confidence,
                'sources': sources_data
            })
            
            # Save to database
            save_chat_history(q, answer, confidence, sources_for_db)
            
        except Exception as e:
            add_backend_log(f"Exception processing question {i+1}: {str(e)}")
            results.append({
                'question': q, 
                'answer': f"Error processing question: {str(e)}", 
                'confidence': 0,
                'sources': {'summary': 'Error occurred', 'detailed': 'Error occurred'}
            })
    
    add_backend_log(f"Completed processing {len(results)} questions")
    
    # Generate CSV file for download with proper formatting for Excel/Sheets
    try:
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_ALL)  # Quote all fields to handle commas in text
        writer.writerow(['Question', 'Answer', 'Source', 'Confidence'])
        
        for row in results:
            try:
                # Clean up the sources field for better CSV formatting
                sources_str = ""
                if isinstance(row['sources'], dict):
                    sources_str = row['sources'].get('summary', 'N/A')
                else:
                    sources_str = str(row['sources'])
                
                # Clean up the answer field - remove newlines and extra spaces
                clean_answer = re.sub(r'\s+', ' ', str(row['answer'])).strip()
                
                # Ensure all fields are strings and handle any encoding issues
                question = str(row['question']).encode('utf-8', errors='ignore').decode('utf-8')
                answer = clean_answer.encode('utf-8', errors='ignore').decode('utf-8')
                source = sources_str.encode('utf-8', errors='ignore').decode('utf-8')
                confidence = str(row['confidence'])
                
                writer.writerow([question, answer, source, confidence])
            except Exception as row_error:
                add_backend_log(f"Error writing CSV row: {row_error}")
                # Write a placeholder row if there's an error
                writer.writerow(['Error processing question', 'Error processing answer', 'N/A', '0'])
        
        output.seek(0)
        add_backend_log(f"CSV generated successfully with {len(results)} rows")
    except Exception as csv_error:
        add_backend_log(f"Error generating CSV: {csv_error}")
        # Create a minimal CSV if generation fails
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_ALL)
        writer.writerow(['Question', 'Answer', 'Source', 'Confidence'])
        writer.writerow(['Error generating CSV', 'Please try again', 'N/A', '0'])
        output.seek(0)
    
    # Create a temporary file for the CSV
    import tempfile
    import os
    import uuid
    
    # Generate a unique filename
    unique_id = str(uuid.uuid4())
    csv_filename = f'answers_{len(results)}_questions_{unique_id}.csv'
    
    # Create temp file with unique name
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    temp_file.write(output.getvalue())
    temp_file.close()
    
    # Store the mapping in a simple in-memory dict (in production, use Redis or database)
    if not hasattr(app, 'temp_files'):
        app.temp_files = {}
    app.temp_files[unique_id] = temp_file.name
    
    # Return both JSON for chat display and CSV file ID
    return jsonify({
        'questions': [r['question'] for r in results],
        'answers': [r['answer'] for r in results],
        'confidences': [r['confidence'] for r in results],
        'sources': [r['sources'] for r in results],
        'csv_id': unique_id,
        'filename': csv_filename
    })

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

@app.route('/get_history', methods=['GET'])
def get_history():
    session_key = request.args.get('session_key')
    history = get_chat_history(session_key=session_key)
    return jsonify({'history': history})

@app.route('/get_documents', methods=['GET'])
def get_documents():
    """Get list of uploaded documents"""
    try:
        docs_dir = 'your_docs'
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
        
        docs_dir = 'your_docs'
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
        docs_dir = 'your_docs'
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
        
        # Use incremental update instead of full rebuild
        global ai_app
        if ai_app and ai_app.db:
            # Database exists, use incremental update
            if ai_app.add_document_incremental(file_path):
                add_backend_log(f"Document uploaded incrementally: {file.filename}")
            else:
                # Fallback to full rebuild if incremental fails
                ai_app.build_db("incremental_update")
                add_backend_log(f"Document uploaded with fallback rebuild: {file.filename}")
        else:
            # No database exists, do initial build
            ai_app = AIApp(None)
            ai_app.build_db("initial_build")
            add_backend_log(f"Document uploaded with initial build: {file.filename}")
        
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
        
        # Always reinitialize AI with updated documents
        global ai_app
        ai_app = AIApp(None)
        ai_app.build_db()
        
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
        else:
            add_backend_log(f"File not found on disk: {file_path}")
        # Delete from the database
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM documents WHERE filename = %s', (filename,))
        conn.commit()
        conn.close()
        # Always reinitialize AI with updated documents
        global ai_app
        ai_app = AIApp(None)
        ai_app.build_db()
        add_backend_log(f"Document deleted: {filename}")
        return jsonify({'status': 'success', 'message': f'Document {filename} deleted successfully'})
    except Exception as e:
        add_backend_log(f"Error message: {e}")
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
        
        # Delete from database
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM documents WHERE filename = %s', (filename,))
        conn.commit()
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
                    for result in successful_uploads:
                        file_path = os.path.join('your_docs', result['filename'])
                        ai_app.add_document_incremental(file_path)
                else:
                    # Do initial build for new database
                    ai_app = AIApp(None)
                    ai_app.build_db("bulk_upload_build")
                ai_app_ready = True
                add_backend_log(f"Database updated after bulk upload of {len(successful_uploads)} documents")
            except Exception as e:
                add_backend_log(f"Error updating database after bulk upload: {e}")
        
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
        docs_dir = 'your_docs'
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

if __name__ == '__main__':
    print('Flask app starting...')
    
    # Initialize database tables
    try:
        init_db()
        print("Database tables initialized successfully")
    except Exception as e:
        print(f"Error initializing database: {e}")
    
    import socket
    host = '127.0.0.1'
    port = 5000
    print(f' * Running on http://{host}:{port}')
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        if local_ip != host:
            print(f' * Running on http://{local_ip}:{port}')
    except Exception as e:
        print(f' * Could not determine local IP: {e}')
    app.run(host=host, port=port, debug=True)