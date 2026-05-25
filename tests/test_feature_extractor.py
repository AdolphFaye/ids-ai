"""
tests/test_feature_extractor.py
───────────────────────────────────────────────────────────────
Tests unitaires — Module FeatureExtractor
Auteur : Alioune Badara Adolphe Faye
"""

import pytest
import numpy as np
import pandas as pd

from src.feature_engineering.feature_extractor import FeatureExtractor


# ─────────────────────────────────────────────────────────────────────────────
#  FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def extractor() -> FeatureExtractor:
    return FeatureExtractor()


@pytest.fixture
def sample_network_df() -> pd.DataFrame:
    """DataFrame simulant des flux réseau CICIDS après prétraitement."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "Src IP":                        [f"192.168.1.{i % 10}" for i in range(n)],
        "Dst IP":                        [f"10.0.0.{i % 5}" for i in range(n)],
        "Protocol":                      np.random.choice([6, 17, 1], n),
        "Flow Duration":                 np.random.randint(100, 100000, n),
        "Total Fwd Packets":             np.random.randint(1, 100, n),
        "Total Backward Packets":        np.random.randint(1, 50, n),
        "Total Length of Fwd Packets":   np.random.randint(100, 10000, n),
        "Total Length of Bwd Packets":   np.random.randint(50, 5000, n),
        "Flow IAT Mean":                 np.random.uniform(1, 100, n),
        "Flow IAT Std":                  np.random.uniform(0.1, 20, n),
        "Flow IAT Max":                  np.random.uniform(50, 500, n),
        "Flow IAT Min":                  np.random.uniform(0.01, 10, n),
        "Label":                         np.random.choice(["BENIGN", "DoS", "PortScan"], n),
    })


# ─────────────────────────────────────────────────────────────────────────────
#  TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestColumnNormalization:

    def test_aliases_resolved(self, extractor, sample_network_df):
        df = extractor._normalize_column_names(sample_network_df.copy())
        assert "src_ip" in df.columns
        assert "dst_ip" in df.columns
        assert "duration" in df.columns
        assert "fwd_packets" in df.columns

    def test_original_alias_removed(self, extractor, sample_network_df):
        df = extractor._normalize_column_names(sample_network_df.copy())
        # Les anciens noms doivent avoir été remplacés
        assert "Src IP" not in df.columns
        assert "Flow Duration" not in df.columns


class TestDerivedFeatures:

    def test_bytes_ratio_added(self, extractor, sample_network_df):
        df = extractor._normalize_column_names(sample_network_df.copy())
        df = extractor._add_derived_features(df)
        assert "bytes_ratio_fwd_bwd" in df.columns

    def test_bytes_ratio_between_0_and_1(self, extractor, sample_network_df):
        df = extractor._normalize_column_names(sample_network_df.copy())
        df = extractor._add_derived_features(df)
        ratio = df["bytes_ratio_fwd_bwd"]
        assert (ratio >= 0).all() and (ratio <= 1).all()

    def test_total_bytes_added(self, extractor, sample_network_df):
        df = extractor._normalize_column_names(sample_network_df.copy())
        df = extractor._add_derived_features(df)
        assert "total_bytes" in df.columns

    def test_iat_range_added(self, extractor, sample_network_df):
        df = extractor._normalize_column_names(sample_network_df.copy())
        df = extractor._add_derived_features(df)
        assert "iat_range" in df.columns

    def test_iat_range_non_negative(self, extractor, sample_network_df):
        df = extractor._normalize_column_names(sample_network_df.copy())
        df = extractor._add_derived_features(df)
        assert (df["iat_range"] >= 0).all()

    def test_flow_bytes_s_computed(self, extractor, sample_network_df):
        df = extractor._normalize_column_names(sample_network_df.copy())
        df = extractor._add_derived_features(df)
        assert "flow_bytes_s" in df.columns


class TestBehavioralFeatures:

    def test_srcip_features_added(self, extractor, sample_network_df):
        df = extractor._normalize_column_names(sample_network_df.copy())
        df = extractor._add_derived_features(df)
        df = extractor._add_behavioral_features(df)
        # Vérifier qu'au moins une feature agrégée src_ip existe
        srcip_feats = [c for c in df.columns if c.startswith("srcip_")]
        assert len(srcip_feats) > 0

    def test_dstip_features_added(self, extractor, sample_network_df):
        df = extractor._normalize_column_names(sample_network_df.copy())
        df = extractor._add_derived_features(df)
        df = extractor._add_behavioral_features(df)
        dstip_feats = [c for c in df.columns if c.startswith("dstip_")]
        assert len(dstip_feats) > 0

    def test_no_nan_in_behavioral_features(self, extractor, sample_network_df):
        df = extractor._normalize_column_names(sample_network_df.copy())
        df = extractor._add_derived_features(df)
        df = extractor._add_behavioral_features(df)
        agg_cols = [c for c in df.columns if c.startswith(("srcip_", "dstip_", "proto_"))]
        assert df[agg_cols].isnull().sum().sum() == 0


class TestExtractAll:

    def test_returns_dataframe(self, extractor, sample_network_df):
        result = extractor.extract_all(sample_network_df.copy(), label_col="Label")
        assert isinstance(result, pd.DataFrame)

    def test_label_preserved(self, extractor, sample_network_df):
        result = extractor.extract_all(sample_network_df.copy(), label_col="Label")
        assert "Label" in result.columns

    def test_more_features_than_input(self, extractor, sample_network_df):
        n_cols_before = sample_network_df.shape[1]
        result = extractor.extract_all(sample_network_df.copy(), label_col="Label")
        assert result.shape[1] > n_cols_before

    def test_same_rows_count(self, extractor, sample_network_df):
        """Le nombre de lignes ne doit pas changer."""
        result = extractor.extract_all(sample_network_df.copy(), label_col="Label")
        assert len(result) == len(sample_network_df)

    def test_no_nan_in_output(self, extractor, sample_network_df):
        result = extractor.extract_all(sample_network_df.copy(), label_col="Label")
        num_cols = result.select_dtypes(include=[np.number]).columns
        assert result[num_cols].isnull().sum().sum() == 0

    def test_get_feature_names(self, extractor, sample_network_df):
        result = extractor.extract_all(sample_network_df.copy(), label_col="Label")
        feat_names = extractor.get_feature_names(result)
        assert "Label" not in feat_names
        assert len(feat_names) > 0
