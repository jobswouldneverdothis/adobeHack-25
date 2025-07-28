FROM python:3.9-slim

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/app/models \
    SENTENCE_TRANSFORMERS_HOME=/app/models \
    HF_HOME=/app/models

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Create startup script
RUN echo '#!/bin/bash\n\
ls -la /app/models/intfloat-multilingual-e5-small\n\
python src/ingest.py\n\
python src/retrieve.py' > /app/run.sh && \
    chmod +x /app/run.sh

ENTRYPOINT ["/bin/bash", "/app/run.sh"]