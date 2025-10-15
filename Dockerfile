# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Create a non-root user and set cache directory permissions
RUN useradd --create-home --shell /bin/bash app && \
    mkdir -p /home/app/.cache && \
    chown -R app:app /home/app/.cache && \
    chown -R app:app /app

# Set environment variables for Hugging Face cache
ENV HF_HOME=/home/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/app/.cache/huggingface/transformers
ENV HF_DATASETS_CACHE=/home/app/.cache/huggingface/datasets

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Change ownership of all files to app user
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Expose the port your app runs on
EXPOSE 8000

# Command to run the application; Hugging Face Spaces sets PORT env
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"