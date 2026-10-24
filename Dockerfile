# syntax=docker/dockerfile:1
# Two small targets from one file:
#   docker build --target api -t kisansense-api .
#   docker build --target app -t kisansense-app .
# Trained models are NOT baked in; mount ./artifacts and ./docs/results (see docker-compose.yml).

FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN useradd --create-home --uid 1000 kisan

# ---------- backend (FastAPI + models) ----------
FROM base AS api
ENV PYTHONPATH=/app:/app/src \
    TF_CPP_MIN_LOG_LEVEL=2
COPY requirements.txt .
# Server needs no notebook/test/UI packages; use the CPU-only TensorFlow build to keep the image smaller.
RUN grep -v -E '^(jupyter|pytest|httpx|streamlit)' requirements.txt \
    | sed 's/^tensorflow>=/tensorflow-cpu>=/' > requirements-api.txt \
    && pip install -r requirements-api.txt
COPY src ./src
COPY backend ./backend
RUN mkdir -p artifacts docs/results && chown -R kisan:kisan /app
USER kisan
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------- frontend (Streamlit, talks to the API over HTTP) ----------
FROM base AS app
ENV KISANSENSE_API_URL=http://api:8000
RUN pip install "streamlit>=1.40" "requests>=2.31" "pandas>=2.1" "pillow>=10.0"
COPY app ./app
COPY .streamlit ./.streamlit
RUN chown -R kisan:kisan /app
USER kisan
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"
CMD ["streamlit", "run", "app/Home.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
