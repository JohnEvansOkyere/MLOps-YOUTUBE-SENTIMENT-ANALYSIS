# Use a fuller Debian-based Python image (better for ML libs)
FROM python:3.11-bullseye

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install system and Python dependencies
RUN apt-get update && apt-get install -y build-essential \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose FastAPI port
EXPOSE 8080

# Start FastAPI server
CMD ["uvicorn", "fastAPI_app.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
