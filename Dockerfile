# Stage 1: Build the React 19 + Vite GIS workstation
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# Stage 2: Production Python runtime with PyTorch CPU & Geospatial Engine
FROM python:3.13-slim AS runner

# Set environment flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    TORCH_NUM_THREADS=4

WORKDIR /app

# Install minimal system dependencies for OpenCV and Rasterio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch directly from official wheel repository (lean ~150MB instead of ~1GB CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user (UID 1000 standard for Hugging Face Spaces)
RUN useradd -m -u 1000 user

# Copy application code and dataset
COPY ml/ ./ml/
COPY dataset/ ./dataset/

# Copy compiled frontend from Stage 1 into frontend/dist
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Ensure writable directories for user uploads and inference outputs
RUN mkdir -p dataset/uploads ml/outputs/inference && \
    chown -R user:user /app

USER user

# Expose port (default 7860 for Hugging Face Spaces)
EXPOSE 7860

# Run FastAPI server
CMD ["sh", "-c", "uvicorn ml.api.server:app --host 0.0.0.0 --port ${PORT:-7860}"]
