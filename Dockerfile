# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Install system dependencies and PostgreSQL
RUN apt-get update && \
    apt-get install -y gcc libpq-dev postgresql postgresql-contrib curl tesseract-ocr sudo procps && \
    rm -rf /var/lib/apt/lists/*

# Find PostgreSQL version and set up proper symlinks
RUN PG_VERSION=$(ls /usr/lib/postgresql/ | head -1) && \
    echo "PostgreSQL version found: $PG_VERSION" && \
    ln -sf /usr/lib/postgresql/$PG_VERSION/bin/* /usr/local/bin/ && \
    echo "export PATH=/usr/lib/postgresql/$PG_VERSION/bin:\$PATH" >> /etc/profile && \
    echo "export PATH=/usr/lib/postgresql/$PG_VERSION/bin:\$PATH" >> /root/.bashrc

# Install Ollama (system-wide)
RUN curl -fsSL https://ollama.com/install.sh | sh

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app && \
    echo "appuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set environment variables (non-sensitive only)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POSTGRES_DB=chat_history
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=321Calvin123
ENV FLASK_APP=main.py

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy only application code (not data files)
COPY main.py .
COPY ai_bot.py .
COPY Logistics_Files/ ./Logistics_Files/
COPY templates/ ./templates/
COPY Docker_Container_Files/ ./Docker_Container_Files/

# Create directories that will be mounted as volumes
RUN mkdir -p your_docs your_db uploads && \
    chown -R appuser:appuser your_docs your_db uploads

# Change ownership of all files to appuser
RUN chown -R appuser:appuser /app

# Make the init script executable
RUN chmod +x Docker_Container_Files/init_postgres.sh

# Create necessary directories for PostgreSQL and set permissions
# Use appuser for all directories since we'll use sudo in the init script
RUN mkdir -p /var/lib/postgresql/data && \
    mkdir -p /var/run/postgresql && \
    chown -R appuser:appuser /var/lib/postgresql && \
    chown -R appuser:appuser /var/run/postgresql

# Expose only the web application port
EXPOSE 5000

# Switch to non-root user
USER appuser

# Initialize the database and run all services
CMD ["./Docker_Container_Files/init_postgres.sh"] 