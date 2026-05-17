"""
detection/model.py
───────────────────
Entraînement et inférence du modèle Isolation Forest.

Isolation Forest – principe :
  • Construit N arbres de décision aléatoires
  • Isole chaque point en le séparant des autres par des coupes
  • Les anomalies s'isolent en moins d'étapes (régions peu denses)
  • Le score d'anomalie est inversement proportionnel
    à la profondeur moyenne d'isolation dans les arbres
  • Seuil de décision : configurable via le paramètre contamination
"""

import numpy as np
from sklearn.ensemble import IsolationForest


def train_model(X_scaled: np.ndarray,
                contamination: float = 0.05,
                n_estimators: int = 100,
                random_state: int = 42) -> IsolationForest:
    """
    Entraîne un modèle Isolation Forest.

    Paramètres
    ----------
    X_scaled      : np.ndarray  – features normalisées
    contamination : float       – proportion estimée d'anomalies (0.0–0.5)
    n_estimators  : int         – nombre d'arbres dans la forêt
    random_state  : int         – graine pour la reproductibilité

    Retourne
    --------
    model : IsolationForest fitted
    """
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples='auto',
        random_state=random_state,
    )
    model.fit(X_scaled)
    return model


def predict(model: IsolationForest,
            X_scaled: np.ndarray):
    """
    Prédit la classe de chaque connexion et calcule son score d'anomalie.

    Isolation Forest retourne :
      +1  → inlier (connexion normale)
      -1  → outlier (anomalie / attaque)

    On convertit en convention cybersécurité :
      0 → Normal
      1 → Attaque

    Paramètres
    ----------
    model    : IsolationForest fitted
    X_scaled : np.ndarray – features normalisées

    Retourne
    --------
    predictions : np.ndarray int  – 0 (normal) ou 1 (attaque)
    scores      : np.ndarray float – score d'anomalie brut
                  (plus négatif = plus suspect)
    """
    raw_preds   = model.predict(X_scaled)
    scores      = model.score_samples(X_scaled)
    predictions = np.where(raw_preds == -1, 1, 0)
    return predictions, scores


def predict_one(model: IsolationForest,
                conn_scaled: np.ndarray):
    """
    Prédit la classe d'une seule connexion normalisée.

    Retourne
    --------
    is_attack : bool
    score     : float
    """
    raw   = model.predict(conn_scaled)[0]
    score = model.score_samples(conn_scaled)[0]
    return raw == -1, score
