#!/bin/bash

# Start Ollama in the background
ollama serve &

# Wait for Ollama to be ready
for i in {1..30}; do
    if curl -s http://localhost:11434 > /dev/null; then
        echo "Ollama is ready!"
        break
    fi
    echo "Waiting for Ollama to be ready... ($i)"
    sleep 2
done

# Pull the Mistral model (if not already present)
ollama pull mistral

# Start PostgreSQL service
service postgresql start
sleep 3

# Set up PostgreSQL user and database
su postgres -c "psql -c \"ALTER USER postgres PASSWORD '321Calvin123';\""
su postgres -c "psql -tc \"SELECT 1 FROM pg_database WHERE datname = 'chat_history';\" | grep -q 1 || psql -c \"CREATE DATABASE chat_history;\""

# Print container IP and access info
CONTAINER_IP=$(hostname -I | awk '{print $1}')
echo "\n==================================================="
echo "App is starting!"
echo "Access from this device:   http://localhost:5000"
echo "Access from other devices: http://$CONTAINER_IP:5000"
echo "==================================================="

# Start Flask application
flask run --host=0.0.0.0 