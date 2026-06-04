FROM nvcr.io/nvidia/pytorch:24.03-py3

# Install system deps
RUN apt-get update && apt-get install -y \
    git curl wget build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Flash attention (compile for H100)
RUN pip install flash-attn --no-build-isolation

WORKDIR /monico
COPY . .

CMD ["python", "scripts/train.py"]
