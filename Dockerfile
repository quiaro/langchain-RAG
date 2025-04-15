FROM node:22 AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm install

COPY frontend ./
RUN npm run build:prod

FROM python:3.11-slim-bullseye

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ /app/

# Create the frontend directory and copy the build files
RUN mkdir -p /app/frontend/build
COPY --from=frontend-builder /app/frontend/build /app/frontend/build

# Verify frontend build exists for debugging
RUN ls -la /app/frontend/build

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=7860

# Expose the port
EXPOSE 7860

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]