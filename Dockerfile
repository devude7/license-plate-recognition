FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

# dependencies
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
  && pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    psycopg[binary] \
    python-dotenv \
    opencv-python \
    numpy \
    pillow \
    ultralytics \
    transformers

COPY src ./src

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
