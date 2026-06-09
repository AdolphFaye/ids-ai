"""
tests/test_model.py
────────────────────
Tests unitaires — Module de modélisation (Isolation Forest).

Couverture :
  • train_model   : entraînement, paramètres, reproductibilité
  • predict       : prédictions binaires, scores d'anomalie
  • predict_one   : inférence sur une connexion unique
"""

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from detection.data_generator import generate_network_logs
from detection.preprocessor    import preprocess
from detection.model           import train_model, predict, predict_one


# ──────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_model_and_data():
    """Pipeline complet : données → normalisation → modèle entraîné."""
    X, y, _ = generate_network_logs(n_normal=500, n_attack=25, seed=42)
    X_scaled, scaler = preprocess(X)
    model = train_model(X_scaled, contamination=0.05, n_estimators=50, random_state=42)
    return model, X_scaled, y, scaler


# ──────────────────────────────────────────────
#  train_model
# ──────────────────────────────────────────────

class TestTrainModel:

    def test_returns_isolation_forest(self, trained_model_and_data):
        """train_model retourne bien un IsolationForest."""
        model, _, _, _ = trained_model_and_data
        assert isinstance(model, IsolationForest)

    def test_model_is_fitted(self, trained_model_and_data):
        """Le modèle est fitted (estimators_ est peuplé)."""
        model, X_scaled, _, _ = trained_model_and_data
        assert hasattr(model, 'estimators_'), "Le modèle doit avoir l'attribut estimators_"
        assert len(model.estimators_) == 50

    def test_contamination_parameter(self):
        """Le paramètre contamination est bien appliqué."""
        X, _, _ = generate_network_logs(n_normal=200, n_attack=10, seed=0)
        X_scaled, _ = preprocess(X)
        model = train_model(X_scaled, contamination=0.10)
        assert model.contamination == 0.10

    def test_reproducibility(self):
        """Même random_state → mêmes prédictions."""
        X, _, _ = generate_network_logs(n_normal=200, n_attack=10, seed=0)
        X_scaled, _ = preprocess(X)
        m1 = train_model(X_scaled, random_state=0)
        m2 = train_model(X_scaled, random_state=0)
        p1, _ = predict(m1, X_scaled)
        p2, _ = predict(m2, X_scaled)
        np.testing.assert_array_equal(p1, p2,
            err_msg="Deux modèles avec le même random_state doivent donner les mêmes prédictions")

    def test_n_estimators(self):
        """Le nombre d'arbres correspond au paramètre n_estimators."""
        X, _, _ = generate_network_logs(n_normal=100, n_attack=5, seed=0)
        X_scaled, _ = preprocess(X)
        model = train_model(X_scaled, n_estimators=30, random_state=0)
        assert len(model.estimators_) == 30


# ──────────────────────────────────────────────
#  predict
# ──────────────────────────────────────────────

class TestPredict:

    def test_returns_two_arrays(self, trained_model_and_data):
        """predict retourne un tuple (predictions, scores)."""
        model, X_scaled, _, _ = trained_model_and_data
        result = predict(model, X_scaled)
        assert isinstance(result, tuple) and len(result) == 2

    def test_predictions_are_binary(self, trained_model_and_data):
        """Les prédictions ne contiennent que 0 (normal) ou 1 (attaque)."""
        model, X_scaled, _, _ = trained_model_and_data
        preds, _ = predict(model, X_scaled)
        unique_vals = set(np.unique(preds))
        assert unique_vals.issubset({0, 1}), \
            f"Valeurs inattendues dans les prédictions : {unique_vals}"

    def test_predictions_shape(self, trained_model_and_data):
        """Le vecteur de prédictions a la même longueur que X."""
        model, X_scaled, _, _ = trained_model_and_data
        preds, scores = predict(model, X_scaled)
        assert preds.shape == (X_scaled.shape[0],)
        assert scores.shape == (X_scaled.shape[0],)

    def test_scores_are_floats(self, trained_model_and_data):
        """Les scores d'anomalie sont des nombres flottants."""
        model, X_scaled, _, _ = trained_model_and_data
        _, scores = predict(model, X_scaled)
        assert scores.dtype in [np.float32, np.float64]

    def test_scores_range(self, trained_model_and_data):
        """Les scores Isolation Forest sont dans l'intervalle [-1, 0]."""
        model, X_scaled, _, _ = trained_model_and_data
        _, scores = predict(model, X_scaled)
        assert scores.min() >= -1.0, "Score minimum attendu ≥ -1.0"
        assert scores.max() <= 0.0, "Score maximum attendu ≤ 0.0"

    def test_attack_scores_lower_than_normal(self, trained_model_and_data):
        """Les connexions d'attaque ont en moyenne des scores plus bas (plus suspects)."""
        model, X_scaled, y, _ = trained_model_and_data
        _, scores = predict(model, X_scaled)
        mean_normal = scores[y == 0].mean()
        mean_attack = scores[y == 1].mean()
        assert mean_attack < mean_normal, \
            "Les attaques doivent avoir des scores d'anomalie plus bas que le trafic normal"

    def test_some_attacks_detected(self, trained_model_and_data):
        """Le modèle doit détecter au moins quelques attaques."""
        model, X_scaled, y, _ = trained_model_and_data
        preds, _ = predict(model, X_scaled)
        detected = ((preds == 1) & (y == 1)).sum()
        assert detected > 0, "Le modèle doit détecter au moins une attaque"

    def test_no_nan_in_scores(self, trained_model_and_data):
        """Aucun NaN dans les scores de sortie."""
        model, X_scaled, _, _ = trained_model_and_data
        _, scores = predict(model, X_scaled)
        assert not np.isnan(scores).any()


# ──────────────────────────────────────────────
#  predict_one
# ──────────────────────────────────────────────

class TestPredictOne:

    def test_returns_tuple(self, trained_model_and_data):
        """predict_one retourne un tuple (label, score)."""
        model, X_scaled, _, _ = trained_model_and_data
        result = predict_one(model, X_scaled[0:1])
        assert isinstance(result, tuple) and len(result) == 2

    def test_label_is_binary(self, trained_model_and_data):
        """Le label retourné est 0 ou 1."""
        model, X_scaled, _, _ = trained_model_and_data
        label, _ = predict_one(model, X_scaled[0:1])
        assert label in (0, 1), f"Label inattendu : {label}"

    def test_score_is_float(self, trained_model_and_data):
        """Le score retourné est un float."""
        model, X_scaled, _, _ = trained_model_and_data
        _, score = predict_one(model, X_scaled[0:1])
        assert isinstance(score, float)

    def test_consistent_with_batch_predict(self, trained_model_and_data):
        """predict_one donne le même résultat que predict sur une seule ligne."""
        model, X_scaled, _, _ = trained_model_and_data
        conn = X_scaled[10:11]
        label_one, score_one  = predict_one(model, conn)
        labels_batch, scores_batch = predict(model, conn)
        assert label_one == labels_batch[0]
        assert abs(score_one - scores_batch[0]) < 1e-9
