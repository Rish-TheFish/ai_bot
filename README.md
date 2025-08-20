# FAQ Bot Web

A web-based AI Compliance Assistant for document Q&A, topic management, and batch processing.

## Features
- In-memory logging (no log files)
- Topic and document management
- Real-time UI updates
- Persistent chat history per topic combination
- Batch Q&A (TXT, CSV, PDF, DOCX)
- OCR support for scanned PDFs
- PostgreSQL and FAISS vector DB

## Setup
1. Clone the repo and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Initialize services:
   ```bash
   ./start_services.sh
   ```
3. Start the app:
   ```bash
   python main.py
   ```

## Docker
Build and run with Docker:
```bash
docker build -t faq-bot-web .
docker run -p 5000:5000 faq-bot-web
```

## Troubleshooting
- If you see database connection errors, check your `config_details.py` password and PostgreSQL status.
- If the vector DB is stale, use the app's document management to trigger a rebuild.
- For OCR, ensure `tesseract-ocr` is installed (Dockerfile does this automatically).

## Security
- No log files are written to disk; logs are in-memory and visible in the UI and terminal.
- All document uploads require at least one topic.

## License
MIT 