"""
tests/test_preprocessor.py
───────────────────────────────────────────────────────────────
Tests unitaires — Module DataPreprocessor
Auteur : Alioune Badara Adolphe Faye
"""

import pytest
import numpy as np
import pandas as pd

from src.preprocessing.data_preprocessor import DataPreprocessor


# ─────────────────────────────────────────────────────────────────────────────
#  FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """DataFrame simple avec colonnes num + catégorielles + NaN + inf."""
    return pd.DataFrame({
        "col_a":    [1.0, 2.0, np.nan, 4.0, 5.0],
        "col_b":    [10.0, np.inf, 30.0, 40.0, 50.0],
        "col_cat":  ["tcp", "udp", "tcp", "icmp", "udp"],
        "Label":    ["BENIGN", "DoS", "BENIGN", "BENIGN", "DoS"],
    })


@pytest.fixture
def preprocessor() -> DataPreprocessor:
    return DataPreprocessor(dataset_type="CICIDS2017")


# ─────────────────────────────────────────────────────────────────────────────
#  TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestDropDuplicates:

    def test_removes_exact_duplicates(self, preprocessor):
        df = pd.DataFrame({
            "a": [1, 1, 2],
            "b": [3, 3, 4],
        })
        result = preprocessor._drop_duplicates(df)
        assert len(result) == 2

    def test_no_duplicates_unchanged(self, preprocessor):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = preprocessor._drop_duplicates(df)
        assert len(result) == 3


class TestRemoveInfinite:

    def test_inf_replaced_by_nan(self, preprocessor):
        df = pd.DataFrame({"a": [1.0, np.inf, -np.inf, 4.0]})
        result = preprocessor._remove_inf(df)
        assert not np.any(np.isinf(result["a"].fillna(0).values))
        assert result["a"].isna().sum() == 2

    def test_no_inf_unchanged(self, preprocessor):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = preprocessor._remove_inf(df)
        assert list(result["a"]) == [1.0, 2.0, 3.0]


class TestFitTransform:

    def test_returns_dataframe(self, preprocessor, sample_df):
        result = preprocessor.fit_transform(sample_df.copy(), label_col="Label")
        assert isinstance(result, pd.DataFrame)

    def test_label_preserved(self, preprocessor, sample_df):
        result = preprocessor.fit_transform(sample_df.copy(), label_col="Label")
        assert "Label" in result.columns

    def test_no_infinite_values_after(self, preprocessor, sample_df):
        result = preprocessor.fit_transform(sample_df.copy(), label_col="Label")
        num_cols = result.select_dtypes(include=[np.number]).columns
        assert not np.any(np.isinf(result[num_cols].fillna(0).values))

    def test_no_nan_in_numeric_after(self, preprocessor, sample_df):
        result = preprocessor.fit_transform(sample_df.copy(), label_col="Label")
        num_cols = result.select_dtypes(include=[np.number]).columns
        assert result[num_cols].isnull().sum().sum() == 0

    def test_categorical_encoded_as_int(self, preprocessor, sample_df):
        result = preprocessor.fit_transform(sample_df.copy(), label_col="Label")
        assert result["col_cat"].dtype in [np.int32, np.int64, int]

    def test_fitted_flag_set(self, preprocessor, sample_df):
        preprocessor.fit_transform(sample_df.copy(), label_col="Label")
        assert preprocessor._fitted is True


class TestTransform:

    def test_transform_before_fit_raises(self, preprocessor, sample_df):
        with pytest.raises(RuntimeError, match="fit_transform"):
            preprocessor.transform(sample_df.copy())

    def test_transform_after_fit(self, preprocessor, sample_df):
        preprocessor.fit_transform(sample_df.copy(), label_col="Label")
        df_test = sample_df.copy()
        result = preprocessor.transform(df_test, label_col="Label")
        assert isinstance(result, pd.DataFrame)

    def test_transform_handles_unseen_categories(self, preprocessor, sample_df):
        """Une catégorie inconnue ne doit pas faire planter le transform."""
        preprocessor.fit_transform(sample_df.copy(), label_col="Label")
        df_test = sample_df.copy()
        df_test["col_cat"] = ["ftp", "ssh", "ftp", "ftp", "ssh"]  # inconnus
        result = preprocessor.transform(df_test, label_col="Label")
        assert isinstance(result, pd.DataFrame)


class TestSummary:

    def test_summary_returns_dict(self, preprocessor, sample_df):
        preprocessor.fit_transform(sample_df.copy(), label_col="Label")
        s = preprocessor.summary(sample_df)
        assert isinstance(s, dict)
        assert "rows" in s
        assert "columns" in s
        assert "memory_mb" in s
