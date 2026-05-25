"""
tests/test_preprocessor.py
───────────────────────────
Tests unitaires — Module de prétraitement.

Couverture :
  • preprocess      : normalisation, scaler retourné, shape conservée
  • transform_one   : normalisation d'une seule connexion
"""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from detection.preprocessor import preprocess, transform_one


# ──────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def sample_data():
    """Matrice de features brutes 50×6 reproductible."""
    np.random.seed(42)
    return np.random.rand(50, 6) * 1000  # échelles variées


# ──────────────────────────────────────────────
#  preprocess
# ──────────────────────────────────────────────

class TestPreprocess:

    def test_returns_tuple(self, sample_data):
        """preprocess retourne un tuple (array, scaler)."""
        result = preprocess(sample_data)
        assert isinstance(result, tuple) and len(result) == 2

    def test_output_shape_preserved(self, sample_data):
        """La shape de X est conservée après normalisation."""
        X_scaled, _ = preprocess(sample_data)
        assert X_scaled.shape == sample_data.shape

    def test_scaler_type(self, sample_data):
        """Le scaler retourné est bien un StandardScaler."""
        _, scaler = preprocess(sample_data)
        assert isinstance(scaler, StandardScaler)

    def test_mean_near_zero(self, sample_data):
        """Après normalisation, chaque feature a une moyenne ≈ 0."""
        X_scaled, _ = preprocess(sample_data)
        means = X_scaled.mean(axis=0)
        np.testing.assert_allclose(means, 0, atol=1e-10,
            err_msg="La moyenne de chaque feature doit être proche de 0")

    def test_std_near_one(self, sample_data):
        """Après normalisation, chaque feature a un écart-type ≈ 1."""
        X_scaled, _ = preprocess(sample_data)
        stds = X_scaled.std(axis=0)
        np.testing.assert_allclose(stds, 1, atol=1e-10,
            err_msg="L'écart-type de chaque feature doit être proche de 1")

    def test_no_nan_output(self, sample_data):
        """Aucun NaN dans les données normalisées."""
        X_scaled, _ = preprocess(sample_data)
        assert not np.isnan(X_scaled).any()

    def test_scaler_is_fitted(self, sample_data):
        """Le scaler peut transformer de nouvelles données sans erreur."""
        _, scaler = preprocess(sample_data)
        new_data = np.random.rand(5, 6) * 1000
        try:
            scaler.transform(new_data)
        except Exception as e:
            pytest.fail(f"Le scaler fitté ne devrait pas lever d'erreur : {e}")

    def test_different_scales_equalized(self):
        """Des features à échelles très différentes sont ramenées à la même."""
        X = np.column_stack([
            np.random.normal(0, 1, 100),        # petite échelle
            np.random.normal(100_000, 50_000, 100),  # grande échelle
        ])
        X_scaled, _ = preprocess(X)
        std_col0 = X_scaled[:, 0].std()
        std_col1 = X_scaled[:, 1].std()
        assert abs(std_col0 - std_col1) < 0.1, \
            "Les deux colonnes doivent avoir des écarts-types similaires après normalisation"


# ──────────────────────────────────────────────
#  transform_one
# ──────────────────────────────────────────────

class TestTransformOne:

    def test_shape_preserved(self, sample_data):
        """transform_one préserve la shape (1, n_features)."""
        _, scaler = preprocess(sample_data)
        conn = np.random.rand(1, 6) * 1000
        result = transform_one(scaler, conn)
        assert result.shape == (1, 6)

    def test_consistent_with_bulk_transform(self, sample_data):
        """transform_one donne le même résultat que scaler.transform."""
        _, scaler = preprocess(sample_data)
        conn = sample_data[0:1]
        result_one  = transform_one(scaler, conn)
        result_bulk = scaler.transform(conn)
        np.testing.assert_array_almost_equal(result_one, result_bulk)

    def test_no_nan_output(self, sample_data):
        """Aucun NaN dans le résultat de transform_one."""
        _, scaler = preprocess(sample_data)
        conn = np.random.rand(1, 6) * 500
        result = transform_one(scaler, conn)
        assert not np.isnan(result).any()
