# AI Compliance Assistant - Web Interface

This is a Flask-based web interface for the AI Compliance Assistant, converted from the original tkinter GUI.

## Features

- **Modern Web Interface**: Beautiful, responsive web UI built with Bootstrap
- **Document Upload**: Upload compliance documents (PDF, DOCX, TXT, CSV, XML)
- **AI Q&A**: Ask questions about your compliance documents
- **Mode Switching**: Toggle between Q&A and Checker modes
- **Chat Export**: Export conversation history
- **S3 Integration**: Optional AWS S3 storage support
- **Security**: PIN-protected document uploads

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have Ollama running with the Mistral model:
```bash
ollama pull mistral
```

## Usage

1. Start the Flask application:
```bash
python main.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Initialize the AI Assistant:
   - Click the "Initialize" button
   - Optionally enable S3 storage and provide credentials
   - Click "Initialize" to start the system

4. Upload Compliance Documents:
   - Click "Upload Docs" button
   - Enter the security PIN (default: 1964)
   - Select your compliance documents
   - Click "Upload" to process them

5. Ask Questions:
   - Type your compliance questions in the input field
   - Press Enter or click "Ask" to get AI responses
   - View confidence scores and source documents

## Configuration

The default PIN for document uploads is `1964`. You can change this in `config_details.py`.

## File Structure

- `main.py` - Flask web application
- `ai_bot.py` - AI backend logic (modified for web interface)
- `terminal_ai.py` - Terminal interface (commented out)
- `config_details.py` - Configuration settings
- `templates/index.html` - Web interface template
- `requirements.txt` - Python dependencies

## Notes

- The terminal AI functionality has been commented out as requested
- All tkinter-specific code has been removed or commented out
- The web interface provides all the same functionality as the original GUI
- The system uses the same AI models and document processing as before

## Troubleshooting

- Make sure Ollama is running with the Mistral model
- Check that all required Python packages are installed
- Ensure the `your_docs` and `your_db` directories exist
- For S3 integration, make sure your AWS credentials are configured 