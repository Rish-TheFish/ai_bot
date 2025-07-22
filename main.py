from flask import Flask, render_template, request, jsonify, session, send_file, redirect, url_for
import os
from ai_bot import AIApp
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
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for session support
ai_app = None

# Database setup - PostgreSQL
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:postGreSQL12#@localhost:5432/chat_history')

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
            SELECT question, answer FROM chat_history 
            WHERE session_id = %s 
            ORDER BY timestamp {order} 
            {limit_clause}
        ''', (session_key,))
    elif user_id:
        cursor.execute(f'''
            SELECT question, answer FROM chat_history 
            WHERE user_id = %s 
            ORDER BY timestamp {order} 
            {limit_clause}
        ''', (user_id,))
    else:
        session_id = get_session_id()
        cursor.execute(f'''
            SELECT question, answer FROM chat_history 
            WHERE session_id = %s 
            ORDER BY timestamp {order} 
            {limit_clause}
        ''', (session_id,))
    history = cursor.fetchall()
    conn.close()
    return [{'question': row['question'], 'answer': row['answer']} for row in history]

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
    cursor.execute('''
        INSERT INTO chat_history (user_id, session_id, question, answer, confidence, sources)
        VALUES (%s, %s, %s, %s, %s, %s)
    ''', (user_id, session_id, question, answer, confidence, sources))
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# Migration function to update existing documents to new schema
def migrate_documents():
    """Migrate existing documents to the new many-to-many topic relationship"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Check if old schema exists (documents table with topic_id column)
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'documents' AND column_name = 'topic_id'
        """)
        
        if cursor.fetchone():
            # Old schema exists, migrate data
            print("🔄 Migrating documents to new schema...")
            
            # Get all documents with their old topic_id
            cursor.execute('SELECT id, filename, topic_id FROM documents WHERE topic_id IS NOT NULL')
            old_docs = cursor.fetchall()
            
            for doc_id, filename, topic_id in old_docs:
                # Insert into new document_topics table
                cursor.execute('''
                    INSERT INTO document_topics (document_id, topic_id)
                    VALUES (%s, %s)
                    ON CONFLICT (document_id, topic_id) DO NOTHING
                ''', (doc_id, topic_id))
            
            # Remove the old topic_id column
            cursor.execute('ALTER TABLE documents DROP COLUMN IF EXISTS topic_id')
            conn.commit()
            print(f"✅ Migrated {len(old_docs)} documents to new schema")
        
        conn.close()
    except Exception as e:
        print(f"⚠️ Migration error: {e}")

# Run migration on startup
migrate_documents()

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
        verification_token = secrets.token_urlsafe(32)
        
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, verification_token) 
            VALUES (%s, %s, %s, %s) RETURNING id
        ''', (username, email, password_hash, verification_token))
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
        return jsonify({
            'status': 'success', 
            'message': 'Registration successful! Please check your email to verify your account.'
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
        return jsonify({'error': str(e)})

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
    global ai_app
    ai_app = AIApp(None)
    doc_status = ai_app.get_document_status()
    return jsonify({'status': 'success', 'message': 'AI Assistant initialized', 'document_status': doc_status})

@app.route('/upload', methods=['POST'])
def upload_docs():
    global ai_app
    if not ai_app:
        return jsonify({'error': 'AI Assistant not initialized'})
    files = request.files.getlist('files')
    temp_paths = []
    for file in files:
        if file.filename:
            temp_path = os.path.join('your_docs', file.filename)
            file.save(temp_path)
            temp_paths.append(temp_path)
    result = ai_app.upload_docs(temp_paths)
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
    global ai_app
    if not ai_app:
        return jsonify({'error': 'AI Assistant not initialized'})
    result = ai_app.get_document_status()
    return jsonify(result)

@app.route('/ask', methods=['POST'])
def ask_question():
    global ai_app
    if not ai_app:
        return jsonify({'error': 'AI Assistant not initialized'})
    data = request.get_json()
    question = data.get('question', '')
    topic_ids = data.get('topic_ids', ['all'])
    session_key = data.get('session_key', None)
    if not question:
        return jsonify({'error': 'Question is required'})
    # Get chat history for this session_key
    chat_history = get_chat_history(session_key=session_key)
    # Call handle_question with chat history and topic_ids
    result = ai_app.handle_question(question, chat_history=chat_history, topic_ids=topic_ids)
    if 'error' in result:
        return jsonify({'error': result['error']})
    # Save to database with session_key
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
        questions = [line.strip() for line in content.splitlines() if line.strip()]
    elif ext == 'pdf':
        file.stream.seek(0)
        pdf = PdfReader(io.BytesIO(file.stream.read()))
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                questions.extend([line.strip() for line in text.splitlines() if line.strip()])
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
    else:
        return jsonify({'error': 'Unsupported file type'})
    # Get answers for each question
    results = []
    for q in questions:
        if not q:
            continue
        result = ai_app.handle_question(q)
        answer = result.get('answer', result.get('error', ''))
        results.append({'question': q, 'answer': answer})
        # Save to database
        save_chat_history(q, answer, result.get('confidence', 0), result.get('sources', ''))
    # Write results to CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Question', 'Answer'])
    for row in results:
        writer.writerow([row['question'], row['answer']])
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='answers.csv'
    )

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
        allowed_extensions = {'.pdf', '.docx', '.txt', '.csv', '.xml'}
        new_file_ext = os.path.splitext(new_filename)[1].lower()
        if new_file_ext not in allowed_extensions:
            return jsonify({'error': f'File type {new_file_ext} not allowed. Allowed types: {", ".join(allowed_extensions)}'})
        
        # Rename the file
        os.rename(old_file_path, new_file_path)
        
        # Wait a moment to ensure file system operations complete
        import time
        time.sleep(0.5)
        
        # Always reinitialize AI with updated documents
        global ai_app
        ai_app = AIApp(None)
        
        return jsonify({'status': 'success', 'message': f'Document renamed from {old_filename} to {new_filename}'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/upload_document', methods=['POST'])
def upload_document():
    """Upload a new document with conflict resolution"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        # Validate file type
        allowed_extensions = {'.pdf', '.docx', '.txt', '.csv', '.xml'}
        if file.filename is None:
            return jsonify({'error': 'Invalid filename'})
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'File type {file_ext} not allowed. Allowed types: {", ".join(allowed_extensions)}'})
        
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
        
        # Always reinitialize AI with new documents
        global ai_app
        ai_app = AIApp(None)
        
        return jsonify({'status': 'success', 'message': f'Document {file.filename} uploaded successfully'})
    except Exception as e:
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
        allowed_extensions = {'.pdf', '.docx', '.txt', '.csv', '.xml'}
        if file.filename is None:
            return jsonify({'error': 'Invalid filename'})
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'File type {file_ext} not allowed. Allowed types: {", ".join(allowed_extensions)}'})
        
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
        
        return jsonify({'status': 'success', 'message': f'Document {old_filename} replaced with {file.filename}'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/delete_document', methods=['POST'])
def delete_document():
    """Delete a document"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'Filename not provided'})
        
        docs_dir = 'your_docs'
        file_path = os.path.join(docs_dir, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'})
        
        # Delete the file
        os.remove(file_path)
        
        # Delete topic relationships from database
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM documents WHERE filename = %s', (filename,))
        conn.commit()
        conn.close()
        
        # Always reinitialize AI with updated documents
        global ai_app
        ai_app = AIApp(None)
        
        return jsonify({'status': 'success', 'message': f'Document {filename} deleted successfully'})
    except Exception as e:
        return jsonify({'error': str(e)})

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
        
        return jsonify({
            'status': 'success', 
            'message': f'Topic "{name}" created successfully',
            'topic_id': topic_id
        })
    except Exception as e:
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
        return jsonify({
            'status': 'success', 
            'message': f'Successfully deleted {deleted_count} topic(s): {", ".join(topic_names)}',
            'deleted_count': deleted_count
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)