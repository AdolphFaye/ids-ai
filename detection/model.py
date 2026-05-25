"""
detection/model.py
───────────────────
Entraînement et inférence des modèles de détection.

Modèles disponibles :
  • Isolation Forest – non supervisé, détecte les anomalies inconnues
  • Random Forest    – supervisé, détecte les attaques connues avec haute précision
"""

import numpy as np
import joblib
from sklearn.ensemble import IsolationForest, RandomForestClassifier

#  ISOLATION FOREST

def train_model(X_scaled: np.ndarray,
                contamination: float = 0.05,
                n_estimators: int = 100,
                random_state: int = 42) -> IsolationForest:
    """
    Entraîne un modèle Isolation Forest.

    Paramètres
    ----------
    X_scaled      : features normalisées
    contamination : proportion estimée d'anomalies (0.0–0.5)
    n_estimators  : nombre d'arbres
    random_state  : graine pour la reproductibilité

    Retourne
    --------
    model : IsolationForest entraîné
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
    Prédit la classe de chaque connexion avec Isolation Forest.

    Retourne
    --------
    predictions : np.ndarray int   – 0 (normal) ou 1 (attaque)
    scores      : np.ndarray float – score d'anomalie brut
    """
    raw_preds   = model.predict(X_scaled)
    scores      = model.score_samples(X_scaled)
    predictions = np.where(raw_preds == -1, 1, 0)
    return predictions, scores


def predict_one(model,
                conn_scaled: np.ndarray):
    """
    Prédit la classe d'une seule connexion.
    Fonctionne avec Isolation Forest ET Random Forest.

    Retourne
    --------
    is_attack : bool
    score     : float
    """
    # Isolation Forest
    if isinstance(model, IsolationForest):
        raw   = model.predict(conn_scaled)[0]
        score = model.score_samples(conn_scaled)[0]
        return raw == -1, score

    # Random Forest
    else:
        proba     = model.predict_proba(conn_scaled)[0][1]
        is_attack = proba >= 0.5
        return is_attack, float(proba)


#  RANDOM FOREST

def train_random_forest(X_scaled: np.ndarray,
                        y_train: np.ndarray,
                        n_estimators: int = 100,
                        random_state: int = 42) -> RandomForestClassifier:
    """
    Entraîne un modèle Random Forest supervisé.

    Paramètres
    ----------
    X_scaled     : features normalisées
    y_train      : labels (0=normal, 1=attaque)
    n_estimators : nombre d'arbres
    random_state : graine pour la reproductibilité

    Retourne
    --------
    model : RandomForestClassifier entraîné
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_scaled, y_train)
    return model


def predict_rf(model: RandomForestClassifier,
               X_scaled: np.ndarray):
    """
    Prédit la classe de chaque connexion avec Random Forest.

    Retourne
    --------
    predictions : np.ndarray int   – 0 (normal) ou 1 (attaque)
    probas      : np.ndarray float – probabilité d'être une attaque
    """
    predictions = model.predict(X_scaled)
    probas      = model.predict_proba(X_scaled)[:, 1]
    return predictions, probas



#  SAUVEGARDE / CHARGEMENT


def save_model(model, path: str):
    """
    Sauvegarde un modèle entraîné sur le disque.

    Paramètres
    ----------
    model : modèle entraîné (IsolationForest ou RandomForest)
    path  : chemin du fichier .pkl
    """
    joblib.dump(model, path)
    print(f"  Modèle sauvegardé : {path}")


def load_model(path: str):
    """
    Charge un modèle depuis le disque.

    Paramètres
    ----------
    path : chemin du fichier .pkl

    Retourne
    --------
    model : modèle chargé prêt à l'emploi
    """
    model = joblib.load(path)
    print(f"  Modèle chargé : {path}")
    return model
