# ============================================================
#  Dockerfile — Space Mission Control RL Environment
#  OpenEnv Hackathon Compliant
# ============================================================

FROM python:3.11-slim

# Metadata
LABEL maintainer="space-mission-rl"
LABEL description="Gymnasium RL Environment for Space Mission Control"

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Working directory
WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY main.py .
COPY env.py .
COPY data_loader.py .
COPY task_builder.py .
COPY evaluator.py .
COPY rewards.py .
COPY test_system.py .
COPY inference.py .

# Optional: copy data files if they exist alongside the Dockerfile
# COPY SpaceEngine_Index.db .
# COPY astrophysical_object_catalog.json .

# OpenEnv Compliance: Default command runs inference.py
CMD ["python", "inference.py"]