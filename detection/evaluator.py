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
    """
    Calcule et affiche les métriques de performance.

    Paramètres
    ----------
    y_true  : labels réels  (0=normal, 1=attaque)
    y_pred  : labels prédits
    scores  : scores d'anomalie bruts (Isolation Forest)
    verbose : afficher le rapport dans la console

    Retourne
    --------
    results : dict avec toutes les métriques
    cm      : matrice de confusion (np.ndarray 2×2)
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # AUC-ROC : on inverse les scores car plus négatif = plus suspect
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
    """Affiche le rapport formaté dans la console."""
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
    """
    Calcule les points de la courbe ROC.

    Retourne
    --------
    fpr : np.ndarray – taux de faux positifs
    tpr : np.ndarray – taux de vrais positifs
    """
    fpr, tpr, _ = roc_curve(y_true, -scores)
    return fpr, tpr
