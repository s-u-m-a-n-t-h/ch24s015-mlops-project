FROM python:3.10-slim

WORKDIR /app

# Use BuildKit cache mount to preserve pip downloads across builds
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade -r requirements.txt

COPY src/app/ ./src/app/

EXPOSE 8501

CMD ["streamlit", "run", "src/app/main.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
