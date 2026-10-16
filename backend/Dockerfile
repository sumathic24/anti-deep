FROM python:3.11-slim

WORKDIR /app

# Install system deps (opencv-python-headless needs glib but not libGL;
# libgl1-mesa-glx no longer exists in current Debian base images)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads models/saved

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
