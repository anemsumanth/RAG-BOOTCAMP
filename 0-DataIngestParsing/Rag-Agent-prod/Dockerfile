# 1️⃣  Builder stage – compile wheels (fast)
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -t /opt/deps

# 2️⃣  Runtime stage – minimal image
FROM python:3.11-slim
WORKDIR /app

# Copy only the needed files
COPY --from=builder /opt/deps /opt/deps
COPY app ./app
COPY --chown=1000:1000 .env .  # keep secrets out of image

# Install deps
ENV PYTHONPATH=/opt/deps
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose
EXPOSE 8000

# Entrypoint
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]