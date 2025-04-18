#!/bin/bash
set -e

echo "Building Docker image..."
docker build -t langchain-rag:latest .

echo "Verifying Docker image file structure..."
# Run a temporary container to verify file structure
docker run --rm langchain-rag:latest ls -la /app/frontend/build
docker run --rm langchain-rag:latest ls -la /app/frontend/build/static

echo "All checks passed! You can now run the container with:"
echo "docker run -p 7860:7860 langchain-rag:latest" 