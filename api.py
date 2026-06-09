"""
api.py
──────
API REST du système de détection d'attaques IDS-IA.

Endpoints :
  GET  /              → Status de l'API
  GET  /health        → Vérifie que les modèles sont chargés
  POST /detect        → Analyse avec Isolation Forest
  POST /detect/rf     → Analyse avec Random Forest
  POST /detect/both   → Analyse avec les 2 modèles + consensus
  POST /simulate      → Simule N connexions temps réel

Lancement :
  pip install fastapi uvicorn
  uvicorn api:app --reload --port 8000

Documentation auto :
  http://localhost:8000/docs
"""

import os
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# ── Imports des fonctions existantes du projet ────────────────
from detection.model         import predict_one, predict, predict_rf
from detection.preprocessor  import transform_one
from detection.data_generator import generate_single_connection

# ── Initialisation FastAPI ────────────────────────────────────
app = FastAPI(
    title="IDS-IA API",
    description="Système de Détection d'Intrusions basé sur IA Comportementale",
    version="2.0.0"
)

# ── Chemins des modèles ───────────────────────────────────────
MODEL_IF_PATH = "detection/model_isolation_forest.pkl"
MODEL_RF_PATH = "detection/model_random_forest.pkl"
SCALER_PATH   = "detection/scaler.pkl"

# ── Chargement des modèles au démarrage ───────────────────────
if_model     = None
rf_model     = None
scaler       = None
models_loaded = False

try:
    if_model = joblib.load(MODEL_IF_PATH)   #  joblib comme dans model.py
    rf_model = joblib.load(MODEL_RF_PATH)
    scaler   = joblib.load(SCALER_PATH)
    models_loaded = True
    print("Modèles chargés avec succès")
except Exception as e:
    print(f" Erreur chargement modèles : {e}")
    print("   → Lance d'abord : python main.py")


# ── Schéma d'entrée : 41 features NSL-KDD ────────────────────
class ConnectionData(BaseModel):
    """Les 41 features du dataset NSL-KDD (déjà encodées)."""
    duration: float                    = Field(0.0,  description="Durée de la connexion")
    protocol_type: int                 = Field(0,    description="Protocole encodé : tcp=0, udp=1, icmp=2")
    service: int                       = Field(0,    description="Service encodé")
    flag: int                          = Field(0,    description="Flag encodé")
    src_bytes: float                   = Field(0.0,  description="Octets envoyés")
    dst_bytes: float                   = Field(0.0,  description="Octets reçus")
    land: int                          = Field(0)
    wrong_fragment: int                = Field(0)
    urgent: int                        = Field(0)
    hot: int                           = Field(0)
    num_failed_logins: int             = Field(0)
    logged_in: int                     = Field(0)
    num_compromised: int               = Field(0)
    root_shell: int                    = Field(0)
    su_attempted: int                  = Field(0)
    num_root: int                      = Field(0)
    num_file_creations: int            = Field(0)
    num_shells: int                    = Field(0)
    num_access_files: int              = Field(0)
    num_outbound_cmds: int             = Field(0)
    is_host_login: int                 = Field(0)
    is_guest_login: int                = Field(0)
    count: float                       = Field(0.0)
    srv_count: float                   = Field(0.0)
    serror_rate: float                 = Field(0.0)
    srv_serror_rate: float             = Field(0.0)
    rerror_rate: float                 = Field(0.0)
    srv_rerror_rate: float             = Field(0.0)
    same_srv_rate: float               = Field(0.0)
    diff_srv_rate: float               = Field(0.0)
    srv_diff_host_rate: float          = Field(0.0)
    dst_host_count: float              = Field(0.0)
    dst_host_srv_count: float          = Field(0.0)
    dst_host_same_srv_rate: float      = Field(0.0)
    dst_host_diff_srv_rate: float      = Field(0.0)
    dst_host_same_src_port_rate: float = Field(0.0)
    dst_host_srv_diff_host_rate: float = Field(0.0)
    dst_host_serror_rate: float        = Field(0.0)
    dst_host_srv_serror_rate: float    = Field(0.0)
    dst_host_rerror_rate: float        = Field(0.0)
    dst_host_srv_rerror_rate: float    = Field(0.0)


# ── Schéma pour la simulation ─────────────────────────────────
class SimulationRequest(BaseModel):
    n_connections: int   = Field(10,  ge=1, le=100, description="Nombre de connexions à simuler")
    attack_ratio: float  = Field(0.2, ge=0.0, le=1.0, description="Proportion d'attaques")
    seed: Optional[int]  = Field(42,  description="Graine aléatoire")


# ── Fonction utilitaire ───────────────────────────────────────
def _check_models():
    """Lève une erreur HTTP 503 si les modèles ne sont pas chargés."""
    if not models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Modèles non chargés. Lance d'abord : python main.py"
        )

def _conn_to_array(conn: ConnectionData) -> np.ndarray:
    """
    Convertit ConnectionData → numpy array normalisé (1, 41).
    Utilise transform_one() du preprocessor.py existant.
    """
    features = np.array([[
        conn.duration, conn.protocol_type, conn.service, conn.flag,
        conn.src_bytes, conn.dst_bytes, conn.land, conn.wrong_fragment,
        conn.urgent, conn.hot, conn.num_failed_logins, conn.logged_in,
        conn.num_compromised, conn.root_shell, conn.su_attempted,
        conn.num_root, conn.num_file_creations, conn.num_shells,
        conn.num_access_files, conn.num_outbound_cmds, conn.is_host_login,
        conn.is_guest_login, conn.count, conn.srv_count, conn.serror_rate,
        conn.srv_serror_rate, conn.rerror_rate, conn.srv_rerror_rate,
        conn.same_srv_rate, conn.diff_srv_rate, conn.srv_diff_host_rate,
        conn.dst_host_count, conn.dst_host_srv_count,
        conn.dst_host_same_srv_rate, conn.dst_host_diff_srv_rate,
        conn.dst_host_same_src_port_rate, conn.dst_host_srv_diff_host_rate,
        conn.dst_host_serror_rate, conn.dst_host_srv_serror_rate,
        conn.dst_host_rerror_rate, conn.dst_host_srv_rerror_rate
    ]])
    #  Utilise transform_one() du preprocessor.py
    return transform_one(scaler, features)


# ═══════════════════════════════════════════════════════════════
#  ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/")
def root():
    """Status général de l'API."""
    return {
        "projet"         : "IDS-IA Comportementale",
        "version"        : "2.0.0",
        "status"         : "en ligne ✅",
        "modeles_charges": models_loaded,
        "endpoints"      : {
            "GET"  : ["/", "/health"],
            "POST" : ["/detect", "/detect/rf", "/detect/both", "/simulate"]
        }
    }


@app.get("/health")
def health():
    """Vérifie que les 3 modèles sont bien chargés."""
    _check_models()
    return {
        "status"          : "ok ✅",
        "isolation_forest": "chargé ✅",
        "random_forest"   : "chargé ✅",
        "scaler"          : "chargé ✅"
    }


@app.post("/detect")
def detect_isolation_forest(conn: ConnectionData):
    """
    Analyse une connexion avec l'Isolation Forest (non supervisé).
    ✅ Utilise predict_one() de model.py
    """
    _check_models()

    X = _conn_to_array(conn)

    #  predict_one() retourne (bool, float) — comme défini dans model.py
    is_attack, score = predict_one(if_model, X)

    return {
        "modele"        : "Isolation Forest",
        "verdict"       : "ATTAQUE 🔴" if is_attack else "NORMAL 🟢",
        "score_anomalie": round(float(score), 4),
        "niveau_risque" : "ÉLEVÉ"   if is_attack else "FAIBLE",
        "alerte"        : bool(is_attack)
    }


@app.post("/detect/rf")
def detect_random_forest(conn: ConnectionData):
    """
    Analyse une connexion avec le Random Forest (supervisé).
    ✅ Utilise predict_one() de model.py
    """
    _check_models()

    X = _conn_to_array(conn)

    # ✅ predict_one() fonctionne aussi avec RandomForest (model.py ligne 52)
    is_attack, proba = predict_one(rf_model, X)

    return {
        "modele"               : "Random Forest",
        "verdict"              : "ATTAQUE 🔴" if is_attack else "NORMAL 🟢",
        "probabilite_attaque"  : round(float(proba), 4),
        "probabilite_normal"   : round(1 - float(proba), 4),
        "niveau_risque"        : "ÉLEVÉ"   if is_attack else "FAIBLE",
        "alerte"               : bool(is_attack)
    }


@app.post("/detect/both")
def detect_both(conn: ConnectionData):
    """
    Analyse avec les 2 modèles + verdict de consensus.
    Niveaux de risque :
      CRITIQUE → les 2 modèles détectent une attaque
      MOYEN    → 1 seul modèle détecte une attaque
      FAIBLE   → aucun modèle ne détecte d'attaque
    """
    _check_models()

    X = _conn_to_array(conn)

    # ✅ predict_one() pour les 2 modèles
    alerte_if, score_if  = predict_one(if_model, X)
    alerte_rf, proba_rf  = predict_one(rf_model, X)

    alerte_finale = alerte_if or alerte_rf

    if alerte_if and alerte_rf:
        niveau_risque = "CRITIQUE 🔴"
    elif alerte_finale:
        niveau_risque = "MOYEN 🟡"
    else:
        niveau_risque = "FAIBLE 🟢"

    return {
        "isolation_forest": {
            "verdict"       : "ATTAQUE" if alerte_if else "NORMAL",
            "score_anomalie": round(float(score_if), 4),
            "alerte"        : bool(alerte_if)
        },
        "random_forest": {
            "verdict"             : "ATTAQUE" if alerte_rf else "NORMAL",
            "probabilite_attaque" : round(float(proba_rf), 4),
            "alerte"              : bool(alerte_rf)
        },
        "consensus": {
            "alerte_finale": bool(alerte_finale),
            "verdict"      : "⚠️ ATTAQUE DÉTECTÉE" if alerte_finale else "✅ CONNEXION NORMALE",
            "niveau_risque": niveau_risque
        }
    }


@app.post("/simulate")
def simulate(req: SimulationRequest):
    """
    Simule N connexions et les analyse avec les 2 modèles.
    ✅ Utilise generate_single_connection() de data_generator.py
    """
    _check_models()

    np.random.seed(req.seed)
    resultats = []
    nb_alertes = 0

    for i in range(req.n_connections):
        # ✅ generate_single_connection() de data_generator.py
        est_attaque = np.random.rand() < req.attack_ratio
        conn_raw    = generate_single_connection(is_attack=est_attaque)

        # ✅ transform_one() du preprocessor.py
        conn_scaled = transform_one(scaler, conn_raw)

        # ✅ predict_one() pour les 2 modèles
        alerte_if, score_if = predict_one(if_model, conn_scaled)
        alerte_rf, proba_rf = predict_one(rf_model, conn_scaled)
        alerte_finale       = alerte_if or alerte_rf

        if alerte_finale:
            nb_alertes += 1

        resultats.append({
            "id"             : i + 1,
            "vraie_classe"   : "ATTAQUE" if est_attaque else "NORMAL",
            "alerte_if"      : bool(alerte_if),
            "alerte_rf"      : bool(alerte_rf),
            "alerte_finale"  : bool(alerte_finale),
            "score_if"       : round(float(score_if), 4),
            "proba_rf"       : round(float(proba_rf), 4),
        })

    return {
        "resume": {
            "total_connexions" : req.n_connections,
            "total_alertes"    : nb_alertes,
            "taux_detection"   : round(nb_alertes / req.n_connections * 100, 1)
        },
        "connexions": resultats
    }