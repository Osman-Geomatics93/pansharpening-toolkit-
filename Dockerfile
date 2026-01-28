# Pansharpening Toolkit Docker Image
# Supports both CPU and GPU (NVIDIA)

# Base image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY pyproject.toml .
COPY setup.py .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    numpy \
    scipy \
    scikit-image \
    matplotlib \
    tqdm \
    rasterio \
    jupyter

# Copy source code
COPY . .

# Install package
RUN pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p /app/data /app/results /app/checkpoints

# Expose Jupyter port
EXPOSE 8888

# Default command
CMD ["python", "-c", "from models import create_model; print('Pansharpening Toolkit ready!')"]

# Labels
LABEL org.opencontainers.image.title="Pansharpening Toolkit"
LABEL org.opencontainers.image.description="Deep learning and classic pansharpening methods"
LABEL org.opencontainers.image.authors="Osman O.A. Ibrahim"
LABEL org.opencontainers.image.source="https://github.com/Osman-Geomatics93/pansharpening-toolkit-"
LABEL org.opencontainers.image.licenses="MIT"
