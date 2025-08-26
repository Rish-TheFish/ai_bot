# Use Python 3.11 for compatibility with current ML libraries
FROM python:3.11-slim

# Set environment variables for optimization
ENV PYTORCH_ENABLE_MPS_FALLBACK=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV BLAS_NUM_THREADS=1
ENV LAPACK_NUM_THREADS=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=random
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    postgresql \
    postgresql-contrib \
    curl \
    tesseract-ocr \
    sudo \
    procps \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libopenblas-dev \
    liblapack-dev \
    libeigen3-dev \
    libffi-dev \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set up PostgreSQL
RUN PG_VERSION=$(ls /usr/lib/postgresql/ | head -1) && \
    ln -sf /usr/lib/postgresql/$PG_VERSION/bin/* /usr/local/bin/ && \
    echo "export PATH=/usr/lib/postgresql/$PG_VERSION/bin:\$PATH" >> /etc/profile && \
    echo "export PATH=/usr/lib/postgresql/$PG_VERSION/bin:\$PATH" >> /root/.bashrc

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app && \
    echo "appuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set database environment variables
ENV POSTGRES_DB=chat_history
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=321Calvin123
ENV FLASK_APP=main.py
ENV FLASK_ENV=production
ENV FLASK_DEBUG=0

# Port configuration for VM deployment
ENV PORT=5000

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with current versions
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch==2.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY ai_bot.py .
COPY Logistics_Files/ ./Logistics_Files/
COPY templates/ ./templates/

# Create necessary directories
RUN mkdir -p your_docs your_db uploads logs cache temp && \
    mkdir -p /var/lib/postgresql/data && \
    mkdir -p /var/run/postgresql

# Set permissions
RUN chown -R appuser:appuser /app && \
    chown -R appuser:appuser /var/lib/postgresql && \
    chown -R appuser:appuser /var/run/postgresql

# System optimizations
RUN echo "vm.max_map_count=262144" >> /etc/sysctl.conf && \
    echo "fs.file-max=65536" >> /etc/sysctl.conf

# Pre-download Ollama model during build
RUN echo "📥 Pre-downloading Ollama model during build..." && \
    mkdir -p /home/appuser/.ollama && \
    chown -R appuser:appuser /home/appuser/.ollama && \
    sudo -u appuser ollama serve & \
    sleep 15 && \
    sudo -u appuser ollama pull cogito:3b && \
    sudo -u appuser ollama list | grep -q "cogito:3b" && \
    echo "✓ Model cogito:3b downloaded successfully" && \
    pkill ollama || true

# Create startup script for VM deployment
RUN echo '#!/bin/bash' > /app/startup.sh && \
    echo 'set -euo pipefail' >> /app/startup.sh && \
    echo '' >> /app/startup.sh && \
    echo 'echo "Starting FAQ Bot services..."' >> /app/startup.sh && \
    echo '' >> /app/startup.sh && \
    echo 'echo "Checking for postgres user..."' >> /app/startup.sh && \
    echo 'id postgres || echo "Creating postgres user..." && useradd -r -s /bin/bash -d /var/lib/postgresql postgres || echo "User already exists"' >> /app/startup.sh && \
    echo '' >> /app/startup.sh && \
    echo '# Initialize PostgreSQL database if not already done' >> /app/startup.sh && \
    echo 'if [ ! -f /var/lib/postgresql/data/PG_VERSION ]; then' >> /app/startup.sh && \
    echo '    echo "Initializing PostgreSQL database..."' >> /app/startup.sh && \
    echo '    sudo -u postgres initdb -D /var/lib/postgresql/data --auth=trust' >> /app/startup.sh && \
    echo '    echo "Database initialized successfully"' >> /app/startup.sh && \
    echo '    echo "Fixing permissions..."' >> /app/startup.sh && \
    echo '    sudo chown -R postgres:postgres /var/lib/postgresql/data' >> /app/startup.sh && \
    echo '    sudo chown -R postgres:postgres /var/run/postgresql' >> /app/startup.sh && \
    echo 'fi' >> /app/startup.sh && \
    echo '' >> /app/startup.sh && \
    echo '# Start PostgreSQL' >> /app/startup.sh && \
    echo 'echo "Starting PostgreSQL..."' >> /app/startup.sh && \
    echo 'echo "Postgres user PATH: $(sudo -u postgres bash -c "source /var/lib/postgresql/.profile && echo \$PATH")"' >> /app/startup.sh && \
    echo 'echo "Postgres user home: $(sudo -u postgres echo $HOME)"' >> /app/startup.sh && \
    echo 'echo "Postgres binary location: $(sudo -u postgres bash -c "source /var/lib/postgresql/.profile && which postgres")"' >> /app/startup.sh && \
    echo 'echo "System postgres binary: $(which postgres)"' >> /app/startup.sh && \
    echo 'echo "System pg_ctl binary: $(which pg_ctl)"' >> /app/startup.sh && \
    echo 'sudo -u postgres bash -c "source /var/lib/postgresql/.profile && pg_ctl -D /var/lib/postgresql/data start > /dev/null 2>&1"' >> /app/startup.sh && \
    echo 'sleep 5' >> /app/startup.sh && \
    echo 'echo "PostgreSQL status:"' >> /app/startup.sh && \
    echo 'if sudo -u postgres bash -c "source /var/lib/postgresql/.profile && pg_ctl -D /var/lib/postgresql/data status"; then' >> /app/startup.sh && \
    echo '    echo "PostgreSQL started successfully with pg_ctl"' >> /app/startup.sh && \
    echo 'else' >> /app/startup.sh && \
    echo '    echo "pg_ctl failed, trying direct postgres start..."' >> /app/startup.sh && \
    echo '    sudo -u postgres bash -c "source /var/lib/postgresql/.profile && postgres -D /var/lib/postgresql/data > /var/log/postgresql.log 2>&1 &"' >> /app/startup.sh && \
    echo '    sleep 10' >> /app/startup.sh && \
    echo '    echo "Trying system PATH fallback..."' >> /app/startup.sh && \
    echo '    sudo -u postgres postgres -D /var/lib/postgresql/data > /var/log/postgresql.log 2>&1 &' >> /app/startup.sh && \
    echo '    sleep 10' >> /app/startup.sh && \
    echo 'fi' >> /app/startup.sh && \
    echo 'echo "PostgreSQL process list:"' >> /app/startup.sh && \
    echo 'ps aux | grep postgres || echo "No postgres processes found"' >> /app/startup.sh && \
    echo 'echo "Creating PostgreSQL socket directory..."' >> /app/startup.sh && \
    echo 'sudo mkdir -p /var/run/postgresql' >> /app/startup.sh && \
    echo 'sudo chown postgres:postgres /var/run/postgresql' >> /app/startup.sh && \
    echo 'echo "Setting postgres user permissions..."' >> /app/startup.sh && \
    echo 'sudo chown -R postgres:postgres /var/lib/postgresql' >> /app/startup.sh && \
    echo 'sudo chown -R postgres:postgres /var/run/postgresql' >> /app/startup.sh && \
    echo 'echo "Setting postgres user PATH..."' >> /app/startup.sh && \
    echo 'echo "Available postgres versions:"' >> /app/startup.sh && \
    echo 'ls -la /usr/lib/postgresql/ || echo "No postgres versions found"' >> /app/startup.sh && \
    echo 'echo "Setting PATH for postgres user..."' >> /app/startup.sh && \
    echo 'sudo -u postgres echo "export PATH=/usr/lib/postgresql/*/bin:\$PATH" >> /var/lib/postgresql/.bashrc' >> /app/startup.sh && \
    echo 'sudo -u postgres echo "export PATH=/usr/lib/postgresql/*/bin:\$PATH" >> /var/lib/postgresql/.profile' >> /app/startup.sh && \
    echo '' >> /app/startup.sh && \
    echo '# Create database and user if they do not exist' >> /app/startup.sh && \
    echo 'echo "Setting up database..."' >> /app/startup.sh && \
    echo 'echo "Creating postgres user..."' >> /app/startup.sh && \
    echo 'sudo -u postgres bash -c "source /var/lib/postgresql/.profile && createuser --superuser --createdb --createrole postgres 2>/dev/null || echo \"User postgres already exists\""' >> /app/startup.sh && \
    echo 'echo "Creating database..."' >> /app/startup.sh && \
    echo 'sudo -u postgres bash -c "source /var/lib/postgresql/.profile && createdb chat_history 2>/dev/null || echo \"Database already exists\""' >> /app/startup.sh && \
    echo 'echo "Setting password..."' >> /app/startup.sh && \
    echo 'sudo -u postgres bash -c "source /var/lib/postgresql/.profile && psql -c \"ALTER USER postgres PASSWORD '\''321Calvin123'\'';\" 2>/dev/null || echo \"Password already set\""' >> /app/startup.sh && \
    echo 'echo "Testing database connection..."' >> /app/startup.sh && \
    echo 'sudo -u postgres bash -c "source /var/lib/postgresql/.profile && psql -d chat_history -c \"SELECT version();\"" || echo "Database connection failed"' >> /app/startup.sh && \
    echo '' >> /app/startup.sh && \
    echo '# Start Ollama in background' >> /app/startup.sh && \
    echo 'echo "Starting Ollama..."' >> /app/startup.sh && \
    echo 'ollama serve > /dev/null 2>&1 &' >> /app/startup.sh && \
    echo '' >> /app/startup.sh && \
    echo '# Wait for Ollama' >> /app/startup.sh && \
    echo 'echo "Waiting for Ollama..."' >> /app/startup.sh && \
    echo 'for i in {1..15}; do' >> /app/startup.sh && \
    echo '    if curl -s http://0.0.0.0:11434 > /dev/null 2>&1; then' >> /app/startup.sh && \
    echo '        echo "Ollama ready"' >> /app/startup.sh && \
    echo '        break' >> /app/startup.sh && \
    echo '    fi' >> /app/startup.sh && \
    echo '    sleep 1' >> /app/startup.sh && \
    echo 'done' >> /app/startup.sh && \
    echo '' >> /app/startup.sh && \
    echo '# Start Flask' >> /app/startup.sh && \
    echo 'echo "Starting Flask..."' >> /app/startup.sh && \
    echo 'FLASK_PORT=${PORT:-5000}' >> /app/startup.sh && \
    echo 'python3 main.py' >> /app/startup.sh && \
    chmod +x /app/startup.sh

# Expose port
EXPOSE 5000

# Switch to non-root user
USER appuser

# Create entrypoint script
RUN echo '#!/bin/bash' > /app/entrypoint.sh && \
    echo 'set -e' >> /app/entrypoint.sh && \
    echo 'trap "echo \"Shutting down gracefully...\"; exit 0" SIGTERM SIGINT' >> /app/entrypoint.sh && \
    echo 'exec "$@"' >> /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Set entrypoint and command
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["/app/startup.sh"] 