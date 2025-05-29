# Optimierte Dockerfile für RunPod - Schnellerer Build
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory early
WORKDIR /app

# System dependencies in one layer
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY handler.py .

# Create models directory
RUN mkdir -p /app/models

# Set environment variables for better performance
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models

# Expose port (if needed)
EXPOSE 8080

# Start the handler
CMD ["python", "-u", "handler.py"]
