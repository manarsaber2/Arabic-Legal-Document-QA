# Build the index first (python src/legal_qa/index.py) so data/index/best exists, then:
#   docker build -t arabic-legal-qa .   &&   docker run -p 8000:8000 --env-file .env arabic-legal-qa
FROM python:3.10-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src HF_HOME=/app/.hf INDEX_DIR=/app/data/index/best

RUN pip install --no-cache-dir "torch==2.14.1+cpu" --extra-index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
# the vector store (embedded Qdrant + chunks + meta) ships inside the image
COPY data/index/best ./data/index/best
# bake the embedding model in, so the container starts offline and fast
RUN python -c "import json; from sentence_transformers import SentenceTransformer as S; S(json.load(open('data/index/best/meta.json'))['embedding_model'])"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
# embedded Qdrant allows ONE process: keep a single worker (scale with replicas, or set QDRANT_URL for multi-worker)
CMD ["uvicorn", "legal_qa.api:app", "--host", "0.0.0.0", "--port", "8000"]