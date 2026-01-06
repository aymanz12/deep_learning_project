# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# -----------------------------------------------------------
# ⚡ Install dependencies with the CPU URL
# We use --extra-index-url to let pip find the "+cpu" versions
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
# -----------------------------------------------------------

# Copy application code
COPY app/ ./app/
COPY models/ ./models/

# Expose port
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Run the application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
