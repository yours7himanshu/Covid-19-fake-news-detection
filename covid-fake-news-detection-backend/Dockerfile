# Multi-stage Dockerfile for fast, reproducible deploys

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for numpy/scipy/sklearn
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app code and models
COPY . .

# Expose for Render/containers
ENV PORT=8000
EXPOSE 8000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "4"]

