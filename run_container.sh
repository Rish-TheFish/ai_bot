#!/bin/bash

# FAQ Bot Container Run Script
# This script shows how to run the container with proper volume mounts

echo "🚀 Starting FAQ Bot Container..."

# Create directories if they don't exist
mkdir -p your_docs your_db uploads

# Build the container
echo "🔨 Building container..."
docker build -f Docker_Container_Files/Dockerfile -t faq-bot:latest .

# Stop and remove existing container if running
echo "🛑 Stopping existing container..."
docker stop faq-bot 2>/dev/null || true
docker rm faq-bot 2>/dev/null || true

# Run the container with volume mounts
echo "▶️  Starting container with volume mounts..."
docker run -d \
  --name faq-bot \
  --restart unless-stopped \
  -p 5000:5000 \
  -p 11434:11434 \
  -v "$(pwd)/your_docs:/app/your_docs" \
  -v "$(pwd)/your_db:/app/your_db" \
  -v "$(pwd)/uploads:/app/uploads" \
  -v postgres_data:/var/lib/postgresql/data \
  -v ollama_data:/root/.ollama \
  -e FLASK_ENV=production \
  -e OLLAMA_HOST=0.0.0.0 \
  faq-bot:latest

echo "✅ Container started!"
echo ""
echo "🌐 FAQ Bot is accessible at: http://localhost:5000"
echo "🤖 Ollama API at: http://localhost:11434"
echo ""
echo "📁 Data directories mounted:"
echo "   - your_docs: $(pwd)/your_docs"
echo "   - your_db: $(pwd)/your_db"
echo "   - uploads: $(pwd)/uploads"
echo ""
echo "📊 To check logs: docker logs faq-bot"
echo "🛑 To stop: docker stop faq-bot" 