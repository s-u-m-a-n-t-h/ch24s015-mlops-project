FROM python:3.10-slim

WORKDIR /app

# Use BuildKit cache mount to preserve pip downloads across builds
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade -r requirements.txt

COPY src/api/ ./src/api/

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
