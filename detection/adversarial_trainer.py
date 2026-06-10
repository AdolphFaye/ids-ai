"""
detection/adversarial_trainer.py
──────────────────────────────────
Adversarial Training — Renforcement des modèles contre les attaques d'évasion.

Stratégies d'évasion simulées :
  1. Bruit gaussien      — perturbation aléatoire des features
  2. Camouflage          — imitation du trafic normal
  3. Slow & Low          — trafic lent à très faible débit
  4. Protocol Flip       — changement de protocole réseau

Auteur : Johannes Hounsa — Responsable Sécurité & Évaluation
"""

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from detection.data_generator import FEATURE_NAMES

# ── Index des features ────────────────────────────────────────────────────────
IDX_DURATION   = FEATURE_NAMES.index("duration")
IDX_BYTES_SENT = FEATURE_NAMES.index("bytes_sent")
IDX_BYTES_RECV = FEATURE_NAMES.index("bytes_recv")
IDX_PACKETS    = FEATURE_NAMES.index("nb_packets")
IDX_PORT       = FEATURE_NAMES.index("dst_port")
IDX_PROTOCOL   = FEATURE_NAMES.index("protocol")


# ══════════════════════════════════════════════════════════════════════════════
#  GÉNÉRATEURS D'EXEMPLES ADVERSARIAUX
# ══════════════════════════════════════════════════════════════════════════════

def generate_gaussian_noise(X_attack: np.ndarray,
                             noise_level: float = 0.1,
                             seed: int = 42) -> np.ndarray:
    """
    Stratégie 1 — Bruit gaussien.
    Perturbe aléatoirement chaque feature de ±noise_level * std.

    Paramètres
    ----------
    X_attack    : exemples d'attaques originaux
    noise_level : intensité du bruit (0.1 = 10 % de l'écart-type)
    seed        : graine aléatoire

    Retourne
    --------
    X_perturbed : np.ndarray — attaques perturbées
    """
    np.random.seed(seed)
    std     = X_attack.std(axis=0)
    noise   = np.random.normal(0, noise_level * std, X_attack.shape)
    return X_attack + noise


def generate_camouflage(X_attack: np.ndarray,
                         X_normal: np.ndarray,
                         ratio: float = 0.7,
                         seed: int = 42) -> np.ndarray:
    """
    Stratégie 2 — Camouflage de trafic.
    Mélange les features d'attaque avec les statistiques du trafic normal
    pour imiter un trafic légitime.

    Paramètres
    ----------
    X_attack : exemples d'attaques originaux
    X_normal : exemples de trafic normal de référence
    ratio    : proportion de camouflage (0.7 = 70 % normal, 30 % attaque)
    seed     : graine aléatoire

    Retourne
    --------
    X_camouflage : np.ndarray — attaques camouflées
    """
    np.random.seed(seed)
    normal_mean = X_normal.mean(axis=0)
    X_camouflage = X_attack.copy().astype(float)

    # Mélanger les features non critiques avec les stats du trafic normal
    for i in [IDX_DURATION, IDX_BYTES_RECV, IDX_PACKETS]:
        X_camouflage[:, i] = (
            ratio * normal_mean[i]
            + (1 - ratio) * X_attack[:, i]
        )

    # Imiter les ports normaux (80, 443)
    normal_ports = [80, 443, 8080]
    X_camouflage[:, IDX_PORT] = np.random.choice(normal_ports, len(X_attack))

    # Passer en TCP comme le trafic normal
    X_camouflage[:, IDX_PROTOCOL] = 1.0

    return X_camouflage


def generate_slow_and_low(n_samples: int = 100,
                           seed: int = 42) -> np.ndarray:
    """
    Stratégie 3 — Slow & Low (Low-and-Slow).
    Génère des attaques à très faible débit, réparties sur une longue durée,
    imitant statistiquement le trafic normal pour éviter la détection.

    C'est la stratégie la plus dangereuse : taux d'évasion initial = 100 %.

    Paramètres
    ----------
    n_samples : nombre d'exemples à générer
    seed      : graine aléatoire

    Retourne
    --------
    X_slow : np.ndarray — attaques slow & low
    """
    np.random.seed(seed)

    X_slow = np.column_stack([
        # Durée longue comme le trafic normal (camouflage temporel)
        np.random.normal(90, 20, n_samples),

        # Très peu de bytes envoyés — discret
        np.random.normal(200, 50, n_samples),

        # Bytes reçus similaires au trafic normal
        np.random.normal(7_000, 2_000, n_samples),

        # Très peu de paquets — lent
        np.random.normal(5, 2, n_samples),

        # Ports normaux pour se fondre dans la masse
        np.random.choice([80, 443], n_samples).astype(float),

        # TCP comme le trafic légitime
        np.ones(n_samples),
    ])

    return X_slow


def generate_protocol_flip(X_attack: np.ndarray,
                            seed: int = 42) -> np.ndarray:
    """
    Stratégie 4 — Protocol Flip.
    Inverse le protocole réseau de chaque attaque (UDP→TCP ou TCP→UDP).

    Paramètres
    ----------
    X_attack : exemples d'attaques originaux
    seed     : graine aléatoire

    Retourne
    --------
    X_flipped : np.ndarray — attaques avec protocole inversé
    """
    np.random.seed(seed)
    X_flipped = X_attack.copy().astype(float)

    # Inverser protocol : 0→1 (UDP→TCP) et 1→0 (TCP→UDP)
    X_flipped[:, IDX_PROTOCOL] = 1.0 - X_flipped[:, IDX_PROTOCOL]

    return X_flipped


# ══════════════════════════════════════════════════════════════════════════════
#  AUGMENTATION DU DATASET
# ══════════════════════════════════════════════════════════════════════════════

def augment_with_adversarial(X_train: np.ndarray,
                              y_train: np.ndarray,
                              X_normal: np.ndarray,
                              seed: int = 42) -> tuple:
    """
    Augmente le dataset d'entraînement avec des exemples adversariaux
    générés par les 4 stratégies d'évasion.

    Paramètres
    ----------
    X_train  : features d'entraînement originales
    y_train  : labels d'entraînement originaux
    X_normal : sous-ensemble de trafic normal (pour le camouflage)
    seed     : graine aléatoire

    Retourne
    --------
    X_augmented : np.ndarray — dataset augmenté
    y_augmented : np.ndarray — labels augmentés
    stats       : dict       — statistiques d'augmentation
    """
    X_attacks = X_train[y_train == 1]
    n_attacks = len(X_attacks)

    if n_attacks == 0:
        return X_train, y_train, {"total_added": 0}

    # Générer les exemples adversariaux
    X_noise     = generate_gaussian_noise(X_attacks, seed=seed)
    X_camo      = generate_camouflage(X_attacks, X_normal, seed=seed)
    X_slow      = generate_slow_and_low(n_samples=n_attacks, seed=seed)
    X_flip      = generate_protocol_flip(X_attacks, seed=seed)

    # Labels : tous sont des attaques (1)
    n_added   = n_attacks * 4
    y_added   = np.ones(n_added, dtype=int)

    # Assembler
    X_adversarial = np.vstack([X_noise, X_camo, X_slow, X_flip])
    X_augmented   = np.vstack([X_train, X_adversarial])
    y_augmented   = np.concatenate([y_train, y_added])

    stats = {
        "original_samples"    : len(X_train),
        "adversarial_added"   : n_added,
        "total_samples"       : len(X_augmented),
        "breakdown": {
            "gaussian_noise"  : n_attacks,
            "camouflage"      : n_attacks,
            "slow_and_low"    : n_attacks,
            "protocol_flip"   : n_attacks,
        }
    }

    return X_augmented, y_augmented, stats


# ══════════════════════════════════════════════════════════════════════════════
#  RÉENTRAÎNEMENT ADVERSARIAL
# ══════════════════════════════════════════════════════════════════════════════

def retrain_with_adversarial(model,
                              X_train: np.ndarray,
                              y_train: np.ndarray,
                              X_normal: np.ndarray,
                              seed: int = 42):
    """
    Réentraîne un modèle Random Forest avec augmentation adversariale.

    Paramètres
    ----------
    model    : RandomForestClassifier original
    X_train  : features d'entraînement
    y_train  : labels d'entraînement
    X_normal : trafic normal de référence (pour camouflage)
    seed     : graine aléatoire

    Retourne
    --------
    model_robust : RandomForestClassifier réentraîné
    stats        : dict — statistiques d'augmentation
    """
    X_aug, y_aug, stats = augment_with_adversarial(
        X_train, y_train, X_normal, seed=seed
    )

    # Récupérer les hyperparamètres du modèle original
    params = model.get_params()
    params["random_state"] = seed

    model_robust = RandomForestClassifier(**params)
    model_robust.fit(X_aug, y_aug)

    return model_robust, stats


def compute_evasion_rate(model,
                          X_adversarial: np.ndarray,
                          scaler=None) -> float:
    """
    Calcule le taux d'évasion : proportion d'attaques non détectées.

    Paramètres
    ----------
    model        : modèle entraîné (IsolationForest ou RandomForest)
    X_adversarial: exemples adversariaux (supposés être des attaques)
    scaler       : StandardScaler optionnel pour normaliser avant prédiction

    Retourne
    --------
    evasion_rate : float dans [0, 1] — 1.0 = évasion totale
    """
    if len(X_adversarial) == 0:
        return 0.0

    X = scaler.transform(X_adversarial) if scaler is not None else X_adversarial

    if isinstance(model, IsolationForest):
        raw_preds   = model.predict(X)
        predictions = np.where(raw_preds == -1, 1, 0)
    else:
        predictions = model.predict(X)

    # Taux d'évasion = proportion d'attaques classées comme normales (0)
    evasion_rate = (predictions == 0).sum() / len(predictions)
    return round(float(evasion_rate), 4)
