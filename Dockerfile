# Use official Python image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

COPY . /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port FastAPI will run on
EXPOSE 5000

# Command to run the application
CMD ["uvicorn", "fastAPI_app.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
