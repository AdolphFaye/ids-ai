"""
tests/test_evaluator.py
────────────────────────
Tests unitaires — Module d'évaluation des métriques.

Couverture :
  • evaluate       : métriques calculées, valeurs bornes, cas parfait/nul
  • get_roc_curve  : courbe ROC, forme des arrays
"""

import numpy as np
import pytest

from detection.evaluator import evaluate, get_roc_curve


# ──────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def perfect_predictions():
    """Prédictions parfaites : y_pred == y_true."""
    y_true  = np.array([0, 0, 0, 1, 1, 1])
    y_pred  = np.array([0, 0, 0, 1, 1, 1])
    scores  = np.array([-0.1, -0.1, -0.1, -0.9, -0.9, -0.9])
    return y_true, y_pred, scores


@pytest.fixture
def mixed_predictions():
    """Prédictions réalistes avec quelques erreurs."""
    np.random.seed(42)
    y_true = np.array([0]*90 + [1]*10)
    y_pred = np.array([0]*88 + [1]*2 + [1]*8 + [0]*2)  # 2 FP, 2 FN
    scores = np.concatenate([
        np.random.uniform(-0.3, -0.1, 90),  # normal : scores hauts
        np.random.uniform(-0.9, -0.5, 10),  # attaque : scores bas
    ])
    return y_true, y_pred, scores


# ──────────────────────────────────────────────
#  evaluate
# ──────────────────────────────────────────────

class TestEvaluate:

    def test_returns_dict_and_matrix(self, mixed_predictions):
        """evaluate retourne bien (dict, ndarray 2×2)."""
        y_true, y_pred, scores = mixed_predictions
        results, cm = evaluate(y_true, y_pred, scores, verbose=False)
        assert isinstance(results, dict)
        assert isinstance(cm, np.ndarray)
        assert cm.shape == (2, 2)

    def test_required_keys_present(self, mixed_predictions):
        """Le dictionnaire contient toutes les clés attendues."""
        y_true, y_pred, scores = mixed_predictions
        results, _ = evaluate(y_true, y_pred, scores, verbose=False)
        for key in ['tp', 'tn', 'fp', 'fn', 'precision', 'recall', 'f1', 'auc_roc']:
            assert key in results, f"Clé manquante : {key}"

    def test_perfect_classifier_metrics(self, perfect_predictions):
        """Un classifieur parfait doit avoir precision=recall=f1=1."""
        y_true, y_pred, scores = perfect_predictions
        results, _ = evaluate(y_true, y_pred, scores, verbose=False)
        assert results['precision'] == 1.0, "Precision parfaite attendue"
        assert results['recall']    == 1.0, "Recall parfait attendu"
        assert results['f1']        == 1.0, "F1 parfait attendu"

    def test_perfect_classifier_auc(self, perfect_predictions):
        """Un classifieur parfait doit avoir un AUC-ROC de 1."""
        y_true, y_pred, scores = perfect_predictions
        results, _ = evaluate(y_true, y_pred, scores, verbose=False)
        assert results['auc_roc'] == 1.0, "AUC-ROC parfait attendu"

    def test_confusion_matrix_sum(self, mixed_predictions):
        """La somme de la matrice de confusion = nombre total de samples."""
        y_true, y_pred, scores = mixed_predictions
        _, cm = evaluate(y_true, y_pred, scores, verbose=False)
        assert cm.sum() == len(y_true)

    def test_tp_tn_fp_fn_sum(self, mixed_predictions):
        """TP + TN + FP + FN = nombre total de samples."""
        y_true, y_pred, scores = mixed_predictions
        results, _ = evaluate(y_true, y_pred, scores, verbose=False)
        total = results['tp'] + results['tn'] + results['fp'] + results['fn']
        assert total == len(y_true)

    def test_precision_range(self, mixed_predictions):
        """La précision est dans [0, 1]."""
        y_true, y_pred, scores = mixed_predictions
        results, _ = evaluate(y_true, y_pred, scores, verbose=False)
        assert 0.0 <= results['precision'] <= 1.0

    def test_recall_range(self, mixed_predictions):
        """Le recall est dans [0, 1]."""
        y_true, y_pred, scores = mixed_predictions
        results, _ = evaluate(y_true, y_pred, scores, verbose=False)
        assert 0.0 <= results['recall'] <= 1.0

    def test_f1_range(self, mixed_predictions):
        """Le F1-score est dans [0, 1]."""
        y_true, y_pred, scores = mixed_predictions
        results, _ = evaluate(y_true, y_pred, scores, verbose=False)
        assert 0.0 <= results['f1'] <= 1.0

    def test_auc_range(self, mixed_predictions):
        """L'AUC-ROC est dans [0, 1]."""
        y_true, y_pred, scores = mixed_predictions
        results, _ = evaluate(y_true, y_pred, scores, verbose=False)
        assert 0.0 <= results['auc_roc'] <= 1.0

    def test_f1_formula(self):
        """F1 = 2*TP / (2*TP + FP + FN) — vérification manuelle."""
        y_true  = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred  = np.array([1, 1, 0, 0, 1, 0, 0, 0])
        scores  = np.array([-0.9, -0.8, -0.4, -0.3, -0.7, -0.2, -0.1, -0.15])
        results, _ = evaluate(y_true, y_pred, scores, verbose=False)
        # TP=2, FP=1, FN=2 → F1 = 2*2 / (2*2+1+2) = 4/7 ≈ 0.571
        expected_f1 = round(4 / 7, 4)
        assert abs(results['f1'] - expected_f1) < 0.001

    def test_no_predicted_attacks(self):
        """Cas limite : aucune attaque détectée → precision et f1 à 0."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])  # aucune alerte
        scores = np.array([-0.2, -0.1, -0.3, -0.4])
        results, _ = evaluate(y_true, y_pred, scores, verbose=False)
        assert results['tp'] == 0
        assert results['fn'] == 2
        assert results['f1'] == 0.0


# ──────────────────────────────────────────────
#  get_roc_curve
# ──────────────────────────────────────────────

class TestGetRocCurve:

    def test_returns_two_arrays(self, mixed_predictions):
        """get_roc_curve retourne (fpr, tpr)."""
        y_true, _, scores = mixed_predictions
        fpr, tpr = get_roc_curve(y_true, scores)
        assert isinstance(fpr, np.ndarray)
        assert isinstance(tpr, np.ndarray)

    def test_same_length(self, mixed_predictions):
        """fpr et tpr ont la même longueur."""
        y_true, _, scores = mixed_predictions
        fpr, tpr = get_roc_curve(y_true, scores)
        assert len(fpr) == len(tpr)

    def test_fpr_starts_at_zero(self, mixed_predictions):
        """La courbe ROC commence à (0, 0)."""
        y_true, _, scores = mixed_predictions
        fpr, tpr = get_roc_curve(y_true, scores)
        assert fpr[0] == 0.0

    def test_fpr_ends_at_one(self, mixed_predictions):
        """La courbe ROC se termine à fpr=1."""
        y_true, _, scores = mixed_predictions
        fpr, tpr = get_roc_curve(y_true, scores)
        assert fpr[-1] == 1.0

    def test_values_in_range(self, mixed_predictions):
        """FPR et TPR sont dans [0, 1]."""
        y_true, _, scores = mixed_predictions
        fpr, tpr = get_roc_curve(y_true, scores)
        assert fpr.min() >= 0.0 and fpr.max() <= 1.0
        assert tpr.min() >= 0.0 and tpr.max() <= 1.0
