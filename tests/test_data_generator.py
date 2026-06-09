"""
tests/test_data_generator.py
─────────────────────────────
Tests unitaires — Module de génération de données.

Couverture :
  • generate_network_logs  : dimensions, labels, reproductibilité
  • generate_single_connection : shape, valeurs, normal vs attaque
"""

import numpy as np
import pandas as pd
import pytest

from detection.data_generator import (
    generate_network_logs,
    generate_single_connection,
    FEATURE_NAMES,
)


# ──────────────────────────────────────────────
#  generate_network_logs
# ──────────────────────────────────────────────

class TestGenerateNetworkLogs:

    def test_output_types(self):
        """Les trois valeurs retournées ont les bons types."""
        X, y, df = generate_network_logs(n_normal=100, n_attack=10)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(df, pd.DataFrame)

    def test_dimensions(self):
        """X et y ont les bonnes dimensions."""
        X, y, df = generate_network_logs(n_normal=200, n_attack=20)
        assert X.shape == (220, 6), "X doit avoir 220 lignes et 6 features"
        assert y.shape == (220,),   "y doit avoir 220 labels"

    def test_label_counts(self):
        """Le nombre de labels 0 et 1 correspond aux paramètres."""
        X, y, df = generate_network_logs(n_normal=300, n_attack=30)
        assert (y == 0).sum() == 300, "300 connexions normales attendues"
        assert (y == 1).sum() == 30,  "30 connexions d'attaque attendues"

    def test_dataframe_columns(self):
        """Le DataFrame contient toutes les colonnes attendues."""
        _, _, df = generate_network_logs(n_normal=50, n_attack=5)
        expected_cols = FEATURE_NAMES + ['label', 'label_name']
        for col in expected_cols:
            assert col in df.columns, f"Colonne manquante : {col}"

    def test_label_name_mapping(self):
        """label_name reflète correctement les valeurs de label."""
        _, _, df = generate_network_logs(n_normal=50, n_attack=10)
        assert set(df['label_name'].unique()) == {'Normal', 'Attaque'}
        assert df.loc[df['label'] == 0, 'label_name'].eq('Normal').all()
        assert df.loc[df['label'] == 1, 'label_name'].eq('Attaque').all()

    def test_reproducibility_with_seed(self):
        """Deux appels avec le même seed produisent des données identiques."""
        X1, y1, _ = generate_network_logs(n_normal=100, n_attack=10, seed=0)
        X2, y2, _ = generate_network_logs(n_normal=100, n_attack=10, seed=0)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_differ(self):
        """Deux seeds différents produisent des données différentes."""
        X1, _, _ = generate_network_logs(n_normal=100, n_attack=10, seed=1)
        X2, _, _ = generate_network_logs(n_normal=100, n_attack=10, seed=2)
        assert not np.array_equal(X1, X2)

    def test_no_nan_values(self):
        """Aucune valeur NaN dans les données générées."""
        X, y, df = generate_network_logs(n_normal=100, n_attack=10)
        assert not np.isnan(X).any(), "X ne doit pas contenir de NaN"
        assert not df[FEATURE_NAMES].isnull().any().any()

    def test_attack_duration_shorter_than_normal(self):
        """Les sessions d'attaque ont en moyenne une durée plus courte."""
        _, _, df = generate_network_logs(n_normal=1000, n_attack=100, seed=42)
        mean_normal = df.loc[df['label'] == 0, 'duration'].mean()
        mean_attack = df.loc[df['label'] == 1, 'duration'].mean()
        assert mean_attack < mean_normal, \
            "Les attaques doivent avoir des sessions plus courtes en moyenne"

    def test_attack_bytes_sent_higher(self):
        """Les attaques envoient en moyenne plus de bytes (exfiltration)."""
        _, _, df = generate_network_logs(n_normal=1000, n_attack=100, seed=42)
        mean_normal = df.loc[df['label'] == 0, 'bytes_sent'].mean()
        mean_attack = df.loc[df['label'] == 1, 'bytes_sent'].mean()
        assert mean_attack > mean_normal


# ──────────────────────────────────────────────
#  generate_single_connection
# ──────────────────────────────────────────────

class TestGenerateSingleConnection:

    def test_shape(self):
        """La connexion retournée a la forme (1, 6)."""
        conn = generate_single_connection()
        assert conn.shape == (1, 6), f"Shape attendu (1,6), obtenu {conn.shape}"

    def test_shape_attack(self):
        """Même shape pour une connexion d'attaque."""
        conn = generate_single_connection(is_attack=True)
        assert conn.shape == (1, 6)

    def test_no_nan(self):
        """Aucune valeur NaN dans la connexion générée."""
        for is_attack in [True, False]:
            conn = generate_single_connection(is_attack=is_attack, seed=0)
            assert not np.isnan(conn).any()

    def test_reproducibility_with_seed(self):
        """Même seed → même connexion."""
        c1 = generate_single_connection(is_attack=False, seed=7)
        c2 = generate_single_connection(is_attack=False, seed=7)
        np.testing.assert_array_equal(c1, c2)

    def test_protocol_normal_is_tcp(self):
        """Une connexion normale doit utiliser TCP (protocol=1)."""
        conn = generate_single_connection(is_attack=False, seed=0)
        assert conn[0, 5] == 1.0, "Protocol TCP (1.0) attendu pour trafic normal"

    def test_protocol_attack_is_udp(self):
        """Une connexion d'attaque doit utiliser UDP (protocol=0)."""
        conn = generate_single_connection(is_attack=True, seed=0)
        assert conn[0, 5] == 0.0, "Protocol UDP (0.0) attendu pour attaque"
