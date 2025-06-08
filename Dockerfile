# Gunakan image Python yang ringan tapi kompatibel dengan TensorFlow
FROM python:3.10-slim

# Set direktori kerja di dalam container
WORKDIR /app

# Install dependencies OS yang dibutuhkan TensorFlow & Pillow
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements.txt dan install dependensi Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh isi project ke dalam image
COPY . .

# Load model saat container dijalankan
ENV PORT=5000

# Jalankan menggunakan Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT"]
