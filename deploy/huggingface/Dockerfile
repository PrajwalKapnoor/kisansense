# Single-container build for Hugging Face Spaces (one port is exposed, so API + app run together).
# The API listens only on 127.0.0.1:8000 inside the container; only Streamlit (7860) is public.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    KISANSENSE_API_URL=http://127.0.0.1:8000

# Hugging Face runs containers as uid 1000
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PYTHONPATH=/home/user/app:/home/user/app/src
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN grep -v -E '^(jupyter|pytest|httpx)' requirements.txt \
    | sed 's/^tensorflow>=/tensorflow-cpu>=/' > requirements-space.txt \
    && pip install -r requirements-space.txt

COPY --chown=user . .

USER user
EXPOSE 7860
CMD ["python", "start.py"]
