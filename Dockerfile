# Multi-stage Dockerfile for gamma-adapt
# Supports both GPU and CPU modes

ARG BASE_IMAGE=tensorflow/tensorflow:2.16.2-gpu
FROM ${BASE_IMAGE}

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
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
