FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.2.0
RUN pip install --no-cache-dir fastapi==0.109.0 uvicorn[standard]==0.27.0 \
    python-multipart==0.0.6 pydantic==2.5.3 pandas==2.2.0 \
    scikit-learn==1.6.1 numpy==1.26.3
COPY *.py .
COPY models/ ./models/
COPY data/ ./data/
EXPOSE 8000
CMD sh -c "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"
