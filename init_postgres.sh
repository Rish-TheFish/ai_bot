#!/bin/bash

echo "🚀 Starting FAQ Bot initialization..."

# Set up PostgreSQL PATH
echo "Setting up PostgreSQL environment..."
export PATH="/usr/local/bin:/usr/lib/postgresql/*/bin:$PATH"

# Find PostgreSQL version and set PATH
PG_VERSION=$(ls /usr/lib/postgresql/ | head -1)
if [ ! -z "$PG_VERSION" ]; then
    export PATH="/usr/lib/postgresql/$PG_VERSION/bin:$PATH"
    echo "PostgreSQL version: $PG_VERSION"
    echo "PostgreSQL binaries path: /usr/lib/postgresql/$PG_VERSION/bin"
else
    echo "ERROR: No PostgreSQL version found"
    exit 1
fi

# Check if PostgreSQL binaries are available
echo "Checking PostgreSQL binaries..."
which initdb || echo "initdb not found in PATH"
which pg_ctl || echo "pg_ctl not found in PATH"
which psql || echo "psql not found in PATH"

# Function to handle shutdown gracefully
cleanup() {
    echo "Shutting down services..."
    # Stop Flask app
    if [ ! -z "$FLASK_PID" ]; then
        kill $FLASK_PID 2>/dev/null
    fi
    # Stop PostgreSQL
    if command -v pg_ctl &> /dev/null; then
        sudo -u postgres pg_ctl -D /var/lib/postgresql/data stop -m fast
    fi
    # Stop Ollama
    pkill ollama 2>/dev/null
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Ensure PostgreSQL directories exist with correct permissions
echo "Setting up PostgreSQL directories..."
sudo mkdir -p /var/lib/postgresql/data
sudo mkdir -p /var/run/postgresql
sudo chown -R postgres:postgres /var/lib/postgresql
sudo chown -R postgres:postgres /var/run/postgresql
sudo chmod 700 /var/lib/postgresql/data
sudo chmod 755 /var/run/postgresql

# Initialize PostgreSQL data directory if it doesn't exist
if [ ! -f /var/lib/postgresql/data/postgresql.conf ]; then
    echo "Initializing PostgreSQL data directory..."
    if command -v initdb &> /dev/null; then
        sudo -u postgres initdb -D /var/lib/postgresql/data --auth=trust
        echo "✓ PostgreSQL initialized successfully"
    else
        echo "ERROR: initdb command not found"
        exit 1
    fi
    
    # Only modify pg_hba.conf for network access (keep postgresql.conf as default)
    echo "Configuring network access..."
    sudo -u postgres sh -c 'echo "host all all 0.0.0.0/0 trust" >> /var/lib/postgresql/data/pg_hba.conf'
    echo "✓ Network access configured"
fi

# Start PostgreSQL service
echo "Starting PostgreSQL..."
if command -v pg_ctl &> /dev/null; then
    echo "Starting with pg_ctl..."
    sudo -u postgres pg_ctl -D /var/lib/postgresql/data -l /var/lib/postgresql/data/postgres.log start
    START_RESULT=$?
    echo "pg_ctl start command completed with exit code: $START_RESULT"
    
    if [ $START_RESULT -ne 0 ]; then
        echo "ERROR: pg_ctl failed to start PostgreSQL"
        echo "PostgreSQL log contents:"
        sudo -u postgres cat /var/lib/postgresql/data/postgres.log 2>/dev/null || echo "Cannot read log file"
        exit 1
    fi
else
    echo "ERROR: pg_ctl command not found"
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

# Verify PostgreSQL is actually running
if ! sudo -u postgres pg_isready -q; then
    echo "ERROR: PostgreSQL failed to start properly"
    echo "Checking PostgreSQL logs:"
    sudo -u postgres cat /var/lib/postgresql/data/postgres.log
    exit 1
fi

# Set up PostgreSQL user and database
echo "Setting up database..."
sudo -u postgres psql -c "ALTER USER postgres PASSWORD '321Calvin123';"
sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname = 'chat_history';" | grep -q 1 || sudo -u postgres psql -c "CREATE DATABASE chat_history;"

# Test database connection
echo "Testing database connection..."
if sudo -u postgres psql -d chat_history -c "SELECT version();" > /dev/null 2>&1; then
    echo "✓ Database connection successful"
else
    echo "✗ Database connection failed"
    exit 1
fi

echo "🎉 PostgreSQL is working perfectly!"

# Start Ollama in the background
echo "Starting Ollama..."
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

# Print container IP and access info
CONTAINER_IP=$(hostname -I | awk '{print $1}')
echo ""
echo "==================================================="
echo "FAQ Bot is starting!"
echo "Access from this device:   http://localhost:5000"
echo "Container internal IP:      http://$CONTAINER_IP:5000"
echo "External access:            http://localhost:5000 (from host machine)"
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

echo "🎉 All services are running!"
echo "FAQ Bot is accessible at http://localhost:5000"
echo "Ollama API at http://localhost:11434"
echo "PostgreSQL at localhost:5432"

# Keep the script running and monitor all services
echo "Monitoring services... Press Ctrl+C to stop."
while true; do
    # Check PostgreSQL
    if ! sudo -u postgres pg_isready -q; then
        echo "ERROR: PostgreSQL stopped unexpectedly"
        exit 1
    fi
    
    # Check Ollama
    if ! curl -s http://localhost:11434 > /dev/null; then
        echo "ERROR: Ollama stopped unexpectedly"
        exit 1
    fi
    
    # Check Flask
    if ! curl -s http://localhost:5000 > /dev/null; then
        echo "ERROR: Flask stopped unexpectedly"
        exit 1
    fi
    
    sleep 30
done 