# Multi-stage Dockerfile for gamma-adapt
# Supports both GPU and CPU modes, compatible with ARM64 (Apple Silicon) and AMD64 (Intel)

ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

# Set working directory
WORKDIR /workspace

# Install system dependencies including build tools for compiling Python packages
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies (TensorFlow will auto-detect architecture)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create data and output directories
RUN mkdir -p /workspace/data /workspace/out

# Set environment variables for configurable paths
ENV DATA_DIR=/workspace/data
ENV OUT_DIR=/workspace/out
ENV PYTHONPATH=/workspace

# Default command
CMD ["/bin/bash"]
