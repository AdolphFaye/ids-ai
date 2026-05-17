"""
detection/data_generator.py
────────────────────────────
Génération et simulation des logs réseau.

Features simulées :
  duration    – durée de session (secondes)
  bytes_sent  – données envoyées (octets)
  bytes_recv  – données reçues (octets)
  nb_packets  – nombre de paquets échangés
  dst_port    – port de destination
  protocol    – protocole réseau (1=TCP, 0=UDP)
"""

import numpy as np
import pandas as pd


FEATURE_NAMES = [
    'duration', 'bytes_sent', 'bytes_recv',
    'nb_packets', 'dst_port', 'protocol'
]


def generate_network_logs(n_normal: int = 1000,
                           n_attack: int = 50,
                           seed: int = 42):
    """
    Simule des journaux de connexions réseau.

    Paramètres
    ----------
    n_normal : int  – nombre de connexions normales
    n_attack : int  – nombre de connexions malveillantes
    seed     : int  – graine aléatoire pour la reproductibilité

    Retourne
    --------
    X  : np.ndarray  – matrice des features (n_samples, 6)
    y  : np.ndarray  – labels (0=normal, 1=attaque)
    df : pd.DataFrame – version lisible avec colonnes nommées
    """
    np.random.seed(seed)

    # ── Trafic NORMAL : sessions longues, ports standards ──
    normal = np.column_stack([
        np.random.normal(100, 30, n_normal),          # duration
        np.random.normal(5_000, 2_000, n_normal),     # bytes_sent
        np.random.normal(8_000, 3_000, n_normal),     # bytes_recv
        np.random.normal(50, 20, n_normal),            # nb_packets
        np.random.choice([80, 443, 8080], n_normal),  # dst_port
        np.ones(n_normal),                             # TCP
    ])

    # ── Trafic ATTAQUE : scans rapides, exfiltration, DDoS ──
    attack = np.column_stack([
        np.random.normal(5, 2, n_attack),              # très courtes sessions
        np.random.normal(50_000, 10_000, n_attack),    # gros volumes
        np.random.normal(100, 50, n_attack),            # peu reçu
        np.random.normal(1_000, 200, n_attack),         # rafale de paquets
        np.random.choice([22, 3389, 445], n_attack),   # SSH, RDP, SMB
        np.zeros(n_attack),                             # UDP
    ])

    X = np.vstack([normal, attack])
    y = np.array([0] * n_normal + [1] * n_attack)

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df['label']      = y
    df['label_name'] = df['label'].map({0: 'Normal', 1: 'Attaque'})

    return X, y, df


def generate_single_connection(is_attack: bool = False,
                                seed: int = None) -> np.ndarray:
    """
    Génère une seule connexion (pour la simulation temps réel).

    Retourne un tableau de shape (1, 6).
    """
    if seed is not None:
        np.random.seed(seed)

    if is_attack:
        conn = [
            np.random.normal(5, 2),
            np.random.normal(50_000, 10_000),
            np.random.normal(100, 50),
            np.random.normal(1_000, 200),
            float(np.random.choice([22, 3389])),
            0.0,
        ]
    else:
        conn = [
            np.random.normal(100, 30),
            np.random.normal(5_000, 2_000),
            np.random.normal(8_000, 3_000),
            np.random.normal(50, 20),
            float(np.random.choice([80, 443])),
            1.0,
        ]

    return np.array([conn])
