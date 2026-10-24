# Single-container build for a Hugging Face Space: FastAPI (internal, port 8000) + Streamlit (public, port 7860).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PYTHONPATH=/app:/app/src \
    KISANSENSE_API_URL=http://127.0.0.1:8000

RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN grep -v -E '^(jupyter|pytest|httpx)' requirements.txt \
    | sed 's/^tensorflow>=/tensorflow-cpu>=/' > requirements-space.txt \
    && pip install -r requirements-space.txt

COPY --chown=user:user . .
USER user
EXPOSE 7860
CMD ["python", "start.py"]
