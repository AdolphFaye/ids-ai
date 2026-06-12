FROM python:3.11-slim

LABEL maintainer="Jean Chris Kemuel Koki Sombo"

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# ✅ Utilise python -m uvicorn au lieu de uvicorn directement
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]