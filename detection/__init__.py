"""
detection/
──────────
Package de détection d'attaques réseau par IA comportementale.
"""

from .data_generator import generate_network_logs, generate_single_connection
from .preprocessor   import preprocess, transform_one
from .model          import (train_model, predict, predict_one,
                             train_random_forest, predict_rf,
                             save_model, load_model)
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
    "train_random_forest",
    "predict_rf",
    "save_model",
    "load_model",
    "evaluate",
    "get_roc_curve",
    "plot_all",
    "simulate_realtime",
]