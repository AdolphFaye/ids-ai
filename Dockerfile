# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — IDS Data Engineering Pipeline
# Auteur : Alioune Badara Adolphe Faye
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

LABEL maintainer="Alioune Badara Adolphe Faye"
LABEL description="IDS Behavioral AI — Data Engineering Pipeline"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Dépendances système minimales (nécessaires pour Scapy & PyArrow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpcap-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY . .

# Créer les dossiers de données
RUN mkdir -p data/raw data/processed data/features logs

# Volume pour les données (montage externe)
VOLUME ["/app/data/raw", "/app/data/processed"]

# Point d'entrée par défaut
ENTRYPOINT ["python", "pipeline.py"]
CMD ["--help"]
