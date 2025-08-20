#!/bin/bash

echo "🚀 Starting FAQ Bot services..."

# Set up PostgreSQL PATH
echo "Setting up PostgreSQL environment..."
export PATH="/usr/local/bin:/usr/lib/postgresql/*/bin:$PATH"

# Find PostgreSQL version and set PATH
PG_VERSION=$(ls /usr/lib/postgresql/ 2>/dev/null | head -1)
if [ ! -z "$PG_VERSION" ]; then
    export PATH="/usr/lib/postgresql/$PG_VERSION/bin:$PATH"
    echo "PostgreSQL version: $PG_VERSION"
    echo "PostgreSQL binaries path: /usr/lib/postgresql/$PG_VERSION/bin"
else
    echo "WARNING: No PostgreSQL version found in /usr/lib/postgresql/"
    echo "Trying alternative locations..."
    # Try common PostgreSQL installation paths
    for pg_path in "/usr/bin" "/usr/local/bin" "/opt/homebrew/bin"; do
        if [ -d "$pg_path" ] && command -v "$pg_path/initdb" >/dev/null 2>&1; then
            export PATH="$pg_path:$PATH"
            echo "Found PostgreSQL in: $pg_path"
            break
        fi
    done
fi

# Check if PostgreSQL binaries are available
echo "Checking PostgreSQL binaries..."
if ! command -v initdb >/dev/null 2>&1; then
    echo "ERROR: initdb not found in PATH"
    echo "Available PostgreSQL tools:"
    find /usr -name "initdb" 2>/dev/null || echo "No initdb found in /usr"
    exit 1
fi

if ! command -v pg_ctl >/dev/null 2>&1; then
    echo "ERROR: pg_ctl not found in PATH"
    exit 1
fi

if ! command -v psql >/dev/null 2>&1; then
    echo "ERROR: psql not found in PATH"
    exit 1
fi

echo "✓ All PostgreSQL binaries found"

# Function to handle shutdown gracefully
cleanup() {
    echo ""
    echo "🛑 Shutting down FAQ Bot services gracefully..."
    
    # Stop Flask app
    if [ ! -z "$FLASK_PID" ] && kill -0 $FLASK_PID 2>/dev/null; then
        echo "Stopping Flask application..."
        kill $FLASK_PID 2>/dev/null
        wait $FLASK_PID 2>/dev/null
    fi
    
    # Stop Ollama
    if [ ! -z "$OLLAMA_PID" ] && kill -0 $OLLAMA_PID 2>/dev/null; then
        echo "Stopping Ollama..."
        kill $OLLAMA_PID 2>/dev/null
        wait $OLLAMA_PID 2>/dev/null
    fi
    
    # Stop PostgreSQL
    echo "Stopping PostgreSQL..."
    if command -v pg_ctl &> /dev/null; then
        sudo -u postgres pg_ctl -D /var/lib/postgresql/data stop -m fast 2>/dev/null
    fi
    
    echo "✅ All services stopped gracefully"
    exit 0
}

# Set up signal handlers for graceful shutdown
trap cleanup SIGTERM SIGINT SIGQUIT

# Ensure PostgreSQL directories exist with correct permissions
echo "Setting up PostgreSQL directories..."
mkdir -p /var/lib/postgresql/data 2>/dev/null || sudo mkdir -p /var/lib/postgresql/data
mkdir -p /var/run/postgresql 2>/dev/null || sudo mkdir -p /var/run/postgresql

# Set ownership - try without sudo first, then with sudo
chown -R postgres:postgres /var/lib/postgresql 2>/dev/null || sudo chown -R postgres:postgres /var/lib/postgresql
chown -R postgres:postgres /var/run/postgresql 2>/dev/null || sudo chown -R postgres:postgres /var/run/postgresql
chmod 700 /var/lib/postgresql/data 2>/dev/null || sudo chmod 700 /var/lib/postgresql/data
chmod 755 /var/run/postgresql 2>/dev/null || sudo chmod 755 /var/run/postgresql

# Initialize PostgreSQL data directory if it doesn't exist
if [ ! -f /var/lib/postgresql/data/postgresql.conf ]; then
    echo "Initializing PostgreSQL data directory..."
    
    # Clean up any existing data (with proper permissions)
    if [ -d /var/lib/postgresql/data ]; then
        sudo rm -rf /var/lib/postgresql/data/* 2>/dev/null || true
    fi
    
    # Create fresh data directory
    sudo mkdir -p /var/lib/postgresql/data
    
    # Set proper ownership for the new directory
    sudo chown -R postgres:postgres /var/lib/postgresql/data
    sudo chmod 700 /var/lib/postgresql/data
    
    # Initialize PostgreSQL with simple trust authentication first
    echo "Running initdb..."
    sudo -u postgres initdb -D /var/lib/postgresql/data --auth=trust
    
    if [ $? -eq 0 ]; then
        echo "✓ PostgreSQL initialized successfully"
    else
        echo "✗ PostgreSQL initialization failed"
        exit 1
    fi
    
    # Configure PostgreSQL for container use
    echo "Configuring PostgreSQL for container use..."
    
    # Update postgresql.conf
    sudo -u postgres tee -a /var/lib/postgresql/data/postgresql.conf > /dev/null << EOF
# Container-specific configuration
listen_addresses = 'localhost'
port = 5432
max_connections = 100
shared_buffers = 128MB
effective_cache_size = 512MB
maintenance_work_mem = 64MB
work_mem = 4MB
EOF
    
    # Configure pg_hba.conf for local access only
    sudo -u postgres tee -a /var/lib/postgresql/data/pg_hba.conf > /dev/null << EOF
# Local connections only
local all all trust
host all all 127.0.0.1/32 trust
host all all ::1/128 trust
EOF
    
    echo "✓ PostgreSQL configured for container use"
fi

# Start PostgreSQL service
echo "Starting PostgreSQL..."
sudo -u postgres pg_ctl -D /var/lib/postgresql/data -l /var/lib/postgresql/data/postgres.log start

if [ $? -eq 0 ]; then
    echo "✓ PostgreSQL started successfully"
else
    echo "✗ PostgreSQL failed to start"
    echo "PostgreSQL log contents:"
    sudo -u postgres cat /var/lib/postgresql/data/postgres.log 2>/dev/null || echo "Cannot read log file"
    exit 1
fi

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
for i in {1..30}; do
    if sudo -u postgres pg_isready -q; then
        echo "✓ PostgreSQL is ready!"
        break
    fi
    echo "Waiting for PostgreSQL to be ready... ($i/30)"
    sleep 2
done

# Create database and set password
echo "Setting up database..."
sudo -u postgres psql -c "CREATE DATABASE chat_history;" 2>/dev/null || echo "Database may already exist"
sudo -u postgres psql -c "ALTER USER postgres PASSWORD '321Calvin123';" 2>/dev/null || echo "Password may already be set"

# Test database connection
echo "Testing database connection..."
if sudo -u postgres psql -d chat_history -c "SELECT version();" > /dev/null 2>&1; then
    echo "✓ Database connection successful"
else
    echo "✗ Database connection failed"
    exit 1
fi

echo "🎉 PostgreSQL is working perfectly!"

# Start Ollama in the background (localhost only for security)
echo "Starting Ollama (localhost only)..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:11434 > /dev/null; then
        echo "✓ Ollama is ready!"
        break
    fi
    echo "Waiting for Ollama to be ready... ($i/30)"
    sleep 2
done

# Verify Ollama is actually running
if ! kill -0 $OLLAMA_PID 2>/dev/null; then
    echo "❌ Ollama failed to start"
    exit 1
fi

# Pull the Llama 3 Instruct model
echo "Pulling Llama 3 Instruct model..."
ollama pull llama3:instruct

# Verify the model is available
echo "Verifying model is ready..."
if ollama list | grep -q "llama3:instruct"; then
    echo "✓ Llama 3 Instruct model is ready"
else
    echo "✗ Failed to load Llama 3 Instruct model"
    exit 1
fi

# Print container access info
CONTAINER_IP=$(hostname -I | awk '{print $1}' 2>/dev/null || echo "localhost")
echo ""
echo "==================================================="
echo "FAQ Bot is starting!"
echo "🔒 SECURITY: Only Flask (port 5000) is exposed"
echo "🌐 Public access: http://$CONTAINER_IP:5000"
echo "🔑 PostgreSQL access: SSH tunnel only"
echo "🔒 Ollama access: Internal container only"
echo "==================================================="
echo ""

# Start Flask application in background and capture PID
echo "Starting Flask application..."
python3 main.py &
FLASK_PID=$!

# Wait for Flask to be ready
echo "Waiting for Flask to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:5000 > /dev/null; then
        echo "✓ Flask application is ready!"
        break
    fi
    echo "Waiting for Flask to be ready... ($i/30)"
    sleep 2
done

# Verify Flask is actually running
if ! kill -0 $FLASK_PID 2>/dev/null; then
    echo "❌ Flask application failed to start"
    exit 1
fi

# Check if Flask is responding
if ! curl -s http://localhost:5000 > /dev/null; then
    echo "❌ Flask application is not responding"
    exit 1
fi

echo "🎉 All services are running!"
echo "FAQ Bot is accessible at http://localhost:5000"

# Display final service status
echo ""
echo "==================================================="
echo "🎯 SERVICE STATUS SUMMARY"
echo "==================================================="
echo "✅ PostgreSQL: Running on localhost:5432"
echo "✅ Ollama: Running on localhost:11434"
echo "✅ Llama 3 Model: Loaded and ready"
echo "✅ Flask App: Running on localhost:5000"
echo "✅ Database: chat_history database ready"
echo "==================================================="
echo ""

# Keep the container running and monitor services
echo "Monitoring services..."
while true; do
    # Check if Flask is still running
    if ! kill -0 $FLASK_PID 2>/dev/null; then
        echo "❌ Flask application stopped unexpectedly"
        exit 1
    fi
    
    # Check if Ollama is still running
    if ! kill -0 $OLLAMA_PID 2>/dev/null; then
        echo "❌ Ollama stopped unexpectedly"
        exit 1
    fi
    
    # Check if PostgreSQL is still running
    if ! sudo -u postgres pg_isready -q; then
        echo "❌ PostgreSQL stopped unexpectedly"
        exit 1
    fi
    
    # Check service health every 30 seconds
    echo "✅ All services healthy - PostgreSQL, Ollama, and Flask running"
    
    # Sleep for a bit before next check
    sleep 30
done 