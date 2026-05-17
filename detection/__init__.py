"""
detection/
──────────
Package de détection d'attaques réseau par IA comportementale.

Modules
-------
data_generator  – simulation des logs réseau
preprocessor    – normalisation des features
model           – Isolation Forest (entraînement + inférence)
evaluator       – métriques de performance
visualizer      – graphiques et dashboard
realtime        – simulation temps réel

Usage rapide
------------
from detection import generate_network_logs, train_model, plot_all
"""

from .data_generator import generate_network_logs, generate_single_connection
from .preprocessor   import preprocess, transform_one
from .model          import train_model, predict, predict_one
from .evaluator      import evaluate, get_roc_curve
from .visualizer     import plot_all
from .realtime       import simulate_realtime

__all__ = [
    "generate_network_logs",
    "generate_single_connection",
    "preprocess",
    "transform_one",
    "train_model",
    "predict",
    "predict_one",
    "evaluate",
    "get_roc_curve",
    "plot_all",
    "simulate_realtime",
]