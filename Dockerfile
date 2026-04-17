
FROM python:3.10-slim

WORKDIR /app

RUN useradd --create-home --shell /bin/bash app && \
    mkdir -p /home/app/.cache && \
    chown -R app:app /home/app/.cache && \
    chown -R app:app /app

ENV HF_HOME=/home/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/app/.cache/huggingface/transformers
ENV HF_DATASETS_CACHE=/home/app/.cache/huggingface/datasets

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chown -R app:app /app

USER app

# Expose the port your app runs on
EXPOSE 8000

# Command to run the application; Hugging Face Spaces sets PORT env
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"
