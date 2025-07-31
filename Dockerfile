# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Install system dependencies and PostgreSQL
RUN apt-get update && \
    apt-get install -y gcc libpq-dev postgresql postgresql-contrib curl && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama (system-wide)
RUN curl -fsSL https://ollama.com/install.sh | sh

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
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Make the init script executable
RUN chmod +x init_postgres.sh

# Expose the ports
EXPOSE 5000 5432 11434

# Initialize the database and run all services
CMD ["./init_postgres.sh"] 