# FAQ Bot Application - In-Depth Architectural Diagram

## System Overview

The FAQ Bot is a sophisticated AI-powered document Q&A system built with a microservices architecture that combines natural language processing, vector search, and web technologies to provide intelligent document-based question answering.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Web Browser / Frontend                                                        │
│  • HTML5 Interface (templates/index.html)                                      │
│  • JavaScript for AJAX calls                                                   │
│  • Session Management                                                          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ HTTP/HTTPS
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              WEB LAYER                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Flask Web Application (main.py)                                               │
│  • REST API Endpoints                                                          │
│  • Session Management                                                          │
│  • Request Routing                                                             │
│  • File Upload Handling                                                        │
│  • Authentication & Authorization                                              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Internal Calls
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              AI CORE LAYER                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│  AI Application (ai_bot.py)                                                    │
│  • Document Processing Pipeline                                                │
│  • Vector Database Management                                                  │
│  • LLM Integration (Ollama)                                                   │
│  • Embedding Generation                                                        │
│  • Semantic Search Engine                                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Data Flow
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│  • PostgreSQL Database (User Management, Chat History)                         │
│  • FAISS Vector Database (Document Embeddings)                                │
│  • File System Storage (Document Files)                                        │
│  • Ollama Model Storage                                                        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Detailed Component Architecture

### 1. Web Layer (Flask Application)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              FLASK APPLICATION                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Main Application (main.py)                                                    │
│  ├── Flask App Configuration                                                   │
│  ├── Database Connection Management                                            │
│  ├── Session Management                                                        │
│  ├── Route Definitions                                                         │
│  │   ├── / (Main Interface)                                                   │
│  │   ├── /upload (Document Upload)                                            │
│  │   ├── /ask (Question Processing)                                           │
│  │   ├── /chat_history (Chat Retrieval)                                       │
│  │   ├── /documents (Document Management)                                      │
│  │   ├── /topics (Topic Management)                                            │
│  │   ├── /export (Data Export)                                                 │
│  │   └── /status (System Health)                                               │
│  ├── Background Task Management                                                │
│  │   ├── Progress Cleanup                                                      │
│  │   ├── Database Initialization                                               │
│  │   └── AI App Initialization                                                 │
│  └── Error Handling & Logging                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Key Features:**
- **Lazy Initialization**: AI components initialize in background for faster startup
- **Shared User System**: Single user system with session management
- **Graceful Degradation**: Continues operation even if some components fail
- **Background Processing**: Non-blocking document processing and database building

### 2. AI Core Layer (AI Application)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              AI APPLICATION                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│  AI Application (ai_bot.py)                                                   │
│  ├── Document Processing Pipeline                                              │
│  │   ├── Multi-format Support (PDF, DOCX, TXT, CSV, XML, XLSX)               │
│  │   ├── Text Extraction & Cleaning                                           │
│  │   ├── Intelligent Chunking                                                 │
│  │   ├── Metadata Extraction                                                  │
│  │   └── Quality Validation                                                   │
│  ├── Vector Database Management                                               │
│  │   ├── FAISS Index Creation & Management                                    │
│  │   ├── Embedding Generation (Sentence Transformers)                         │
│  │   ├── Similarity Search Engine                                             │
│  │   ├── Index Persistence & Recovery                                         │
│  │   └── Memory Optimization                                                  │
│  ├── LLM Integration                                                          │
│  │   ├── Ollama Local Model Server                                            │
│  │   ├── Model: cogito:3b (3B parameter model)                               │
│  │   ├── Context-Aware Response Generation                                    │
│  │   ├── Source Attribution                                                   │
│  │   └── Confidence Scoring                                                   │
│  ├── Search & Retrieval Engine                                                │
│  │   ├── Semantic Similarity Search                                           │
│  │   ├── Enhanced Context Filtering                                           │
│  │   ├── Multi-pass Retrieval                                                 │
│  │   ├── Relevance Ranking                                                    │
│  │   └── Source Aggregation                                                   │
│  └── Advanced Features                                                        │
│      ├── Table Extraction (Pandas, Tabula, Camelot)                          │
│      ├── OCR Processing (Tesseract)                                           │
│      ├── Image Analysis (OpenCV)                                              │
│      ├── Multi-language Support                                               │
│      └── Performance Monitoring                                               │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Key Features:**
- **Multi-Model Fallbacks**: Multiple PDF processors for reliability
- **Intelligent Chunking**: Context-aware document segmentation
- **Advanced Table Extraction**: Computer vision-based table recognition
- **Memory Optimization**: Adaptive processing based on available resources
- **Real-time Processing**: Background document processing with progress tracking

### 3. Data Layer

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA STORAGE                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│  PostgreSQL Database                                                           │
│  ├── Users Table                                                               │
│  │   ├── id (Primary Key)                                                     │
│  │   ├── username (Unique)                                                    │
│  │   ├── email (Unique)                                                       │
│  │   ├── password_hash                                                        │
│  │   └── created_at                                                           │
│  ├── Chat History Table                                                       │
│  │   ├── id (Primary Key)                                                     │
│  │   ├── user_id (Foreign Key)                                                │
│  │   ├── session_id                                                           │
│  │   ├── question                                                             │
│  │   ├── answer                                                               │
│  │   ├── sources                                                              │
│  │   └── timestamp                                                            │
│  ├── Documents Table                                                           │
│  │   ├── id (Primary Key)                                                     │
│  │   ├── filename (Unique)                                                    │
│  │   ├── file_path                                                            │
│  │   ├── file_size                                                            │
│  │   ├── file_type                                                            │
│  │   └── uploaded_at                                                          │
│  ├── Topics Table                                                              │
│  │   ├── id (Primary Key)                                                     │
│  │   ├── name (Unique)                                                        │
│  │   ├── description                                                          │
│  │   └── created_at                                                           │
│  └── Document Topics Junction Table                                           │
│      ├── id (Primary Key)                                                     │
│      ├── document_id (Foreign Key)                                            │
│      └── topic_id (Foreign Key)                                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│  FAISS Vector Database                                                         │
│  ├── Document Chunk Embeddings                                                │
│  ├── HNSW Index (Hierarchical Navigable Small World)                          │
│  ├── Similarity Search Algorithms                                             │
│  ├── Index Persistence (.faiss, .pkl files)                                   │
│  └── Memory-Mapped Storage                                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│  File System Storage                                                           │
│  ├── Document Repository (./your_docs/)                                       │
│  ├── Vector Database Files (./your_db/)                                       │
│  ├── Upload Directory (./uploads/)                                            │
│  ├── Cache Directory (./cache/)                                               │
│  └── Logs Directory (./logs/)                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Ollama Model Storage                                                          │
│  ├── Local Model Files                                                        │
│  ├── Model Configuration                                                       │
│  └── Model Cache                                                              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 4. Infrastructure Layer

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              INFRASTRUCTURE                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Docker Containerization                                                       │
│  ├── Multi-stage Build Process                                                │
│  ├── Python 3.11 Base Image                                                   │
│  ├── System Dependencies Installation                                         │
│  │   ├── PostgreSQL Client & Server                                           │
│  │   ├── OCR Tools (Tesseract)                                                │
│  │   ├── Computer Vision Libraries (OpenCV)                                   │
│  │   ├── PDF Processing Tools                                                 │
│  │   └── System Optimization Libraries                                        │
│  ├── Python Dependencies (requirements.txt)                                   │
│  ├── Application Code Deployment                                              │
│  ├── Non-root User Security                                                   │
│  └── Health Checks & Monitoring                                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Docker Compose Orchestration                                                  │
│  ├── Service Definition (faq-bot)                                             │
│  ├── Volume Management                                                         │
│  │   ├── Persistent Data Storage                                              │
│  │   ├── Model Cache Management                                               │
│  │   └── Log & Cache Volumes                                                  │
│  ├── Resource Limits & Reservations                                           │
│  │   ├── Memory: 8GB limit, 4GB reservation                                  │
│  │   └── CPU: 4 cores limit, 2 cores reservation                             │
│  ├── Health Check Configuration                                                │
│  ├── Network Configuration                                                     │
│  └── Environment Variable Management                                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│  System Optimizations                                                          │
│  ├── Thread Management                                                        │
│  │   ├── OMP_NUM_THREADS=1                                                   │
│  │   ├── MKL_NUM_THREADS=1                                                   │
│  │   ├── NUMEXPR_NUM_THREADS=1                                               │
│  │   └── OPENBLAS_NUM_THREADS=1                                              │
│  ├── Memory Management                                                        │
│  │   ├── vm.max_map_count=262144                                              │
│  │   └── fs.file-max=65536                                                    │
│  ├── GPU Acceleration Support                                                 │
│  │   ├── Apple Silicon (MPS) Support                                          │
│  │   ├── CUDA Fallback Support                                                │
│  │   └── CPU Optimization                                                     │
│  └── Process Management                                                       │
│      ├── Background Task Scheduling                                           │
│      ├── Resource Monitoring (psutil)                                         │
│      └── Graceful Shutdown Handling                                           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Architecture

### 1. Document Upload & Processing Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Client    │    │   Flask     │    │   AI App    │    │   Storage   │
│   Upload    │───▶│   Upload    │───▶│   Process   │───▶│   Save      │
│             │    │   Handler   │    │   Pipeline  │    │   Files     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                              │
                                                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Vector    │    │   Embedding │    │   Chunking │    │   Text      │
│   Database  │◀───│   Generation│◀───│   Engine   │◀───│   Extraction│
│   Update    │    │             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### 2. Question Answering Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   User      │    │   Flask     │    │   AI App    │    │   Vector    │
│   Question  │───▶│   Question  │───▶│   Query     │───▶│   Search    │
│             │    │   Handler   │    │   Processor │    │   Engine    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                              │
                                                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Response  │    │   LLM       │    │   Context   │    │   Relevant  │
│   Generation│◀───│   Processing│◀───│   Assembly  │◀───│   Chunks    │
│             │    │             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### 3. Background Processing Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Startup   │    │   Background│    │   Document  │    │   Database  │
│   Trigger   │───▶│   Thread    │───▶│   Processing│───▶│   Building  │
│             │    │   Creation  │    │   Queue     │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                              │
                                                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Progress  │    │   Status    │    │   Vector    │    │   Index     │
│   Tracking  │◀───│   Updates   │◀───│   Database  │◀───│   Persistence│
│             │    │             │    │   Creation  │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## Security Architecture

### 1. Authentication & Authorization
- **Shared User System**: Single user account for simplified access
- **Session Management**: Secure session handling with HTTP-only cookies
- **File Upload Security**: Secure filename handling and validation
- **Input Validation**: Comprehensive input sanitization and validation

### 2. Data Security
- **Database Security**: PostgreSQL with secure connection handling
- **File System Security**: Non-root user execution in containers
- **Network Security**: Isolated Docker network with controlled access
- **Environment Variables**: Secure configuration management

### 3. Container Security
- **Non-root User**: Application runs as non-privileged user
- **Resource Limits**: Memory and CPU constraints to prevent abuse
- **Health Checks**: Regular system health monitoring
- **Graceful Shutdown**: Proper cleanup on container termination

## Performance Architecture

### 1. Optimization Strategies
- **Lazy Initialization**: Components initialize on-demand
- **Background Processing**: Non-blocking document processing
- **Memory Management**: Adaptive processing based on available resources
- **Caching**: Vector database and model caching for faster access

### 2. Scalability Features
- **Stateless Design**: Stateless API endpoints for horizontal scaling
- **Resource Monitoring**: Real-time system resource tracking
- **Background Tasks**: Asynchronous processing for better responsiveness
- **Connection Pooling**: Database connection management

### 3. Monitoring & Health
- **Health Check Endpoints**: System status monitoring
- **Progress Tracking**: Real-time processing progress updates
- **Error Logging**: Comprehensive error tracking and reporting
- **Performance Metrics**: Resource usage and response time monitoring

## Deployment Architecture

### 1. Container Strategy
- **Single Container**: All services in one container for simplicity
- **Multi-service**: PostgreSQL, Ollama, and Flask in single container
- **Volume Mounts**: Persistent data storage across deployments
- **Environment Configuration**: Flexible configuration via environment variables

### 2. Cloud Deployment
- **Google Cloud Run**: Optimized for serverless deployment
- **Docker Support**: Full containerization support
- **Auto-scaling**: Automatic scaling based on demand
- **Health Monitoring**: Built-in health check support

### 3. Local Development
- **Docker Compose**: Local development environment
- **Volume Mounts**: Live code editing and testing
- **Hot Reload**: Development server with auto-reload
- **Debug Mode**: Development-specific configurations

## Technology Stack Summary

### Core Technologies
- **Web Framework**: Flask 3.1.1
- **AI Framework**: LangChain 0.3.27
- **Vector Database**: FAISS 1.11.0
- **Language Model**: Ollama with cogito:3b
- **Embeddings**: Sentence Transformers 2.5.1
- **Database**: PostgreSQL 15+
- **Containerization**: Docker & Docker Compose

### Document Processing
- **PDF**: PyMuPDF, PyPDF2, pypdf
- **Word**: python-docx
- **Text**: Built-in text processing
- **Tables**: Pandas, Tabula, Camelot
- **OCR**: Tesseract with OpenCV

### System Integration
- **Process Management**: psutil, threading
- **File Operations**: pathlib, shutil
- **Data Serialization**: pickle, json
- **HTTP Client**: requests
- **System Monitoring**: Built-in health checks

This architecture provides a robust, scalable, and maintainable foundation for the FAQ Bot application, with clear separation of concerns, comprehensive error handling, and optimized performance characteristics. 