"""
detection/preprocessor.py
──────────────────────────
Normalisation et prétraitement des features réseau.

La normalisation est indispensable pour l'Isolation Forest :
sans elle, les features avec de grandes valeurs absolues
(bytes_sent : ~50 000) dominent celles à petite échelle
(protocol : 0 ou 1), faussant les distances et le modèle.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess(X: np.ndarray):
    """
    Centre et réduit les features (moyenne=0, écart-type=1).

    Paramètres
    ----------
    X : np.ndarray – matrice brute des features

    Retourne
    --------
    X_scaled : np.ndarray  – features normalisées
    scaler   : StandardScaler – objet fitted (pour transformer
               de nouvelles connexions avec .transform())
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def transform_one(scaler: StandardScaler,
                  conn: np.ndarray) -> np.ndarray:
    """
    Normalise une seule connexion en utilisant le scaler déjà fitté.

    Paramètres
    ----------
    scaler : StandardScaler – scaler entraîné sur les données d'entraînement
    conn   : np.ndarray de shape (1, n_features)

    Retourne
    --------
    np.ndarray de shape (1, n_features) normalisé
    """
    return scaler.transform(conn)
