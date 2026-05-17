"""
detection/realtime.py
──────────────────────
Simulation de l'analyse de connexions en temps réel.

Reproduit le comportement d'un système de détection
en production : chaque connexion entrante est analysée
individuellement, et une alerte est générée si le modèle
la classe comme anormale.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from detection.data_generator import generate_single_connection
from detection.preprocessor   import transform_one
from detection.model          import predict_one


def simulate_realtime(model: IsolationForest,
                      scaler: StandardScaler,
                      n_connections: int = 20,
                      attack_ratio: float = 0.2,
                      seed: int = 99):
    """
    Simule l'analyse de n_connections connexions en flux continu.

    Paramètres
    ----------
    model         : IsolationForest entraîné
    scaler        : StandardScaler fitté sur les données d'entraînement
    n_connections : nombre de connexions à simuler
    attack_ratio  : proportion de connexions malveillantes (0.0–1.0)
    seed          : graine aléatoire

    Affiche chaque connexion avec son verdict et son score.
    """
    np.random.seed(seed)

    _print_header()
    alerts = 0

    for i in range(n_connections):
        is_attack = np.random.rand() < attack_ratio
        conn      = generate_single_connection(is_attack=is_attack)
        conn_scaled = transform_one(scaler, conn)

        detected, score = predict_one(model, conn_scaled)

        verdict = "🔴 ALERTE" if detected else "🟢 Normal"
        if detected:
            alerts += 1

        _print_row(i + 1, conn[0], verdict, score)

    _print_footer(n_connections, alerts)


def _print_header():
    sep = "=" * 55
    print(f"\n{sep}")
    print("   SIMULATION TEMPS RÉEL")
    print(sep)
    print(f"{'#':<4} {'Duration':>8} {'Bytes':>8} {'Pkts':>6} "
          f"{'Port':>6} {'Verdict':<15} {'Score':>8}")
    print("-" * 55)


def _print_row(idx: int, conn: np.ndarray,
               verdict: str, score: float):
    print(f"{idx:<4} {conn[0]:>8.1f} {conn[1]:>8.0f} {conn[3]:>6.0f} "
          f"{conn[4]:>6.0f} {verdict:<15} {score:>8.3f}")


def _print_footer(total: int, alerts: int):
    print("-" * 55)
    print(f"  Connexions analysées : {total}")
    print(f"  Alertes déclenchées  : {alerts}")
    print("=" * 55)
