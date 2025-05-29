FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application files
COPY . .

# Create models directory
RUN mkdir -p /app/models

# Download WAN 2.1 model at build time (optional for faster startup)
# RUN python -c "from diffusers import StableVideoDiffusionPipeline; StableVideoDiffusionPipeline.from_pretrained('stabilityai/stable-video-diffusion-img2vid-xt', torch_dtype=torch.float16).save_pretrained('/app/models/wan21')"

# Start the handler
CMD ["python", "-u", "handler.py"]
