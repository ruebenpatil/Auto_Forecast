# Use an official Python base image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv properly
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH manually
ENV PATH="/root/.local/bin:$PATH"

# Clone the GitHub repository
COPY . .

# Install Python dependencies with uv
RUN uv pip install --system .

# RUN uv pip install tensorflow
# Expose FastAPI port
EXPOSE 80

# Run the FastAPI app
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:80"]

