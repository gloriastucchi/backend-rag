FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

# system deps you need (add/remove as needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
  && rm -rf /var/lib/apt/lists/*

# install python deps first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest
COPY . .

EXPOSE 8000
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]
