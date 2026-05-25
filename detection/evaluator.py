"""
detection/evaluator.py
───────────────────────
Évaluation des performances du modèle de détection.

Métriques utilisées :
  • Precision  – parmi les alertes déclenchées, combien sont réelles ?
  • Recall     – parmi les vraies attaques, combien sont détectées ?
  • F1-Score   – moyenne harmonique précision/recall
  • AUC-ROC    – capacité à distinguer normal vs attaque (1.0 = parfait)
  • Matrice de confusion – TP, TN, FP, FN
"""

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def evaluate(y_true: np.ndarray,
             y_pred: np.ndarray,
             scores: np.ndarray,
             verbose: bool = True):

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Si scores sont des probabilités RF (entre 0 et 1), pas besoin d'inverser
    if scores.min() >= 0:
        auc = roc_auc_score(y_true, scores)
    else:
        auc = roc_auc_score(y_true, -scores)

    results = {
        'tp'        : int(tp),
        'tn'        : int(tn),
        'fp'        : int(fp),
        'fn'        : int(fn),
        'auc_roc'   : round(auc, 4),
        'precision' : round(tp / (tp + fp) if (tp + fp) > 0 else 0, 4),
        'recall'    : round(tp / (tp + fn) if (tp + fn) > 0 else 0, 4),
        'f1'        : round(2*tp / (2*tp + fp + fn) if (2*tp+fp+fn) > 0 else 0, 4),
    }

    if verbose:
        _print_report(y_true, y_pred, results)

    return results, cm


def _print_report(y_true, y_pred, results: dict):
    sep = "=" * 55
    print(f"\n{sep}")
    print("   RAPPORT D'ÉVALUATION DU MODÈLE")
    print(sep)
    print(classification_report(y_true, y_pred,
                                 target_names=['Normal', 'Attaque']))
    print(f"  Vrais positifs  (attaques détectées) : {results['tp']}")
    print(f"  Faux positifs   (fausses alertes)    : {results['fp']}")
    print(f"  Vrais négatifs  (normal confirmé)    : {results['tn']}")
    print(f"  Faux négatifs   (attaques manquées)  : {results['fn']}")
    print(f"\n  AUC-ROC : {results['auc_roc']:.3f}  "
          f"(1.0 = parfait, 0.5 = aléatoire)")
    print(sep)


def get_roc_curve(y_true: np.ndarray, scores: np.ndarray):
    if scores.min() >= 0:
        fpr, tpr, _ = roc_curve(y_true, scores)
    else:
        fpr, tpr, _ = roc_curve(y_true, -scores)
    return fpr, tpr