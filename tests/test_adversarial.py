"""
tests/test_adversarial.py
──────────────────────────
Tests unitaires — Robustesse adversariale du système IDS.

Couverture :
  • 4 stratégies d'évasion : bruit, camouflage, slow & low, protocol flip
  • Augmentation du dataset
  • Réentraînement adversarial (Random Forest robuste)
  • Comparaison avant/après : le modèle robuste doit mieux détecter

Auteur : Johannes Hounsa — Responsable Sécurité & Évaluation
"""

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from detection.data_generator import generate_network_logs, FEATURE_NAMES
from detection.adversarial_trainer import (
    generate_gaussian_noise,
    generate_camouflage,
    generate_slow_and_low,
    generate_protocol_flip,
    augment_with_adversarial,
    retrain_with_adversarial,
    compute_evasion_rate,
)

IDX_PROTOCOL = FEATURE_NAMES.index("protocol")
IDX_PORT     = FEATURE_NAMES.index("dst_port")


# ══════════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def dataset():
    """Dataset de base : 500 normaux, 50 attaques."""
    X, y, _ = generate_network_logs(n_normal=500, n_attack=50, seed=42)
    return X, y


@pytest.fixture(scope="module")
def X_attacks(dataset):
    X, y = dataset
    return X[y == 1]


@pytest.fixture(scope="module")
def X_normal(dataset):
    X, y = dataset
    return X[y == 0]


@pytest.fixture(scope="module")
def trained_rf(dataset):
    """Random Forest entraîné sur le dataset de base."""
    X, y = dataset
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model   = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)
    return model, scaler


@pytest.fixture(scope="module")
def trained_if(dataset):
    """Isolation Forest entraîné sur le dataset de base."""
    X, y = dataset
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model    = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(X_scaled)
    return model, scaler


# ══════════════════════════════════════════════════════════════════════════════
#  1. GÉNÉRATION DES EXEMPLES ADVERSARIAUX
# ══════════════════════════════════════════════════════════════════════════════

class TestGaussianNoise:

    def test_same_shape(self, X_attacks):
        X_perturbed = generate_gaussian_noise(X_attacks)
        assert X_perturbed.shape == X_attacks.shape

    def test_values_differ(self, X_attacks):
        """Le bruit doit modifier les valeurs."""
        X_perturbed = generate_gaussian_noise(X_attacks, noise_level=0.1)
        assert not np.allclose(X_perturbed, X_attacks)

    def test_reproducible_with_seed(self, X_attacks):
        X1 = generate_gaussian_noise(X_attacks, seed=0)
        X2 = generate_gaussian_noise(X_attacks, seed=0)
        np.testing.assert_array_equal(X1, X2)

    def test_zero_noise_unchanged(self, X_attacks):
        """Bruit nul = pas de modification."""
        X_perturbed = generate_gaussian_noise(X_attacks, noise_level=0.0)
        np.testing.assert_array_almost_equal(X_perturbed, X_attacks)


class TestCamouflage:

    def test_same_shape(self, X_attacks, X_normal):
        X_camo = generate_camouflage(X_attacks, X_normal)
        assert X_camo.shape == X_attacks.shape

    def test_ports_are_normal(self, X_attacks, X_normal):
        """Les ports doivent être dans les ports normaux : 80, 443, 8080."""
        X_camo = generate_camouflage(X_attacks, X_normal)
        ports  = X_camo[:, IDX_PORT]
        assert all(p in [80, 443, 8080] for p in ports)

    def test_protocol_is_tcp(self, X_attacks, X_normal):
        """Le protocole doit être TCP (1.0) après camouflage."""
        X_camo = generate_camouflage(X_attacks, X_normal)
        assert np.all(X_camo[:, IDX_PROTOCOL] == 1.0)

    def test_different_from_original(self, X_attacks, X_normal):
        X_camo = generate_camouflage(X_attacks, X_normal)
        assert not np.allclose(X_camo, X_attacks)


class TestSlowAndLow:

    def test_correct_shape(self):
        X_slow = generate_slow_and_low(n_samples=50)
        assert X_slow.shape == (50, len(FEATURE_NAMES))

    def test_protocol_is_tcp(self):
        """Slow & Low utilise TCP pour se fondre dans la masse."""
        X_slow = generate_slow_and_low(n_samples=100)
        assert np.all(X_slow[:, IDX_PROTOCOL] == 1.0)

    def test_low_packet_count(self):
        """Les attaques slow & low ont un faible nombre de paquets."""
        IDX_PACKETS = FEATURE_NAMES.index("nb_packets")
        X_slow = generate_slow_and_low(n_samples=200, seed=42)
        mean_packets = X_slow[:, IDX_PACKETS].mean()
        assert mean_packets < 20, f"Trop de paquets : {mean_packets:.1f} (attendu < 20)"

    def test_low_bytes_sent(self):
        """Les attaques slow & low envoient très peu de données."""
        IDX_BYTES_SENT = FEATURE_NAMES.index("bytes_sent")
        X_slow = generate_slow_and_low(n_samples=200, seed=42)
        mean_bytes = X_slow[:, IDX_BYTES_SENT].mean()
        assert mean_bytes < 1000, f"Trop de bytes : {mean_bytes:.0f} (attendu < 1000)"

    def test_reproducible(self):
        X1 = generate_slow_and_low(50, seed=7)
        X2 = generate_slow_and_low(50, seed=7)
        np.testing.assert_array_equal(X1, X2)


class TestProtocolFlip:

    def test_same_shape(self, X_attacks):
        X_flip = generate_protocol_flip(X_attacks)
        assert X_flip.shape == X_attacks.shape

    def test_protocol_inverted(self, X_attacks):
        """Chaque protocole doit être inversé : 0→1 ou 1→0."""
        X_flip = generate_protocol_flip(X_attacks)
        original_proto = X_attacks[:, IDX_PROTOCOL]
        flipped_proto  = X_flip[:, IDX_PROTOCOL]
        np.testing.assert_array_almost_equal(
            original_proto + flipped_proto,
            np.ones(len(X_attacks))
        )

    def test_only_protocol_changed(self, X_attacks):
        """Seule la feature protocole doit changer."""
        X_flip = generate_protocol_flip(X_attacks)
        for i, feat in enumerate(FEATURE_NAMES):
            if feat == "protocol":
                continue
            np.testing.assert_array_almost_equal(
                X_flip[:, i], X_attacks[:, i],
                err_msg=f"Feature '{feat}' ne devrait pas changer"
            )


# ══════════════════════════════════════════════════════════════════════════════
#  2. AUGMENTATION DU DATASET
# ══════════════════════════════════════════════════════════════════════════════

class TestAugmentation:

    def test_augmented_larger_than_original(self, dataset, X_normal):
        X, y = dataset
        X_aug, y_aug, stats = augment_with_adversarial(X, y, X_normal)
        assert len(X_aug) > len(X)
        assert len(y_aug) > len(y)

    def test_labels_consistent(self, dataset, X_normal):
        X, y = dataset
        X_aug, y_aug, _ = augment_with_adversarial(X, y, X_normal)
        assert len(X_aug) == len(y_aug)

    def test_original_preserved(self, dataset, X_normal):
        """Les données originales doivent rester intactes."""
        X, y = dataset
        X_aug, y_aug, _ = augment_with_adversarial(X, y, X_normal)
        np.testing.assert_array_equal(X_aug[:len(X)], X)
        np.testing.assert_array_equal(y_aug[:len(y)], y)

    def test_stats_breakdown(self, dataset, X_normal):
        X, y = dataset
        _, _, stats = augment_with_adversarial(X, y, X_normal)
        n_attacks = (y == 1).sum()
        assert stats["breakdown"]["gaussian_noise"] == n_attacks
        assert stats["breakdown"]["slow_and_low"]   == n_attacks
        assert stats["adversarial_added"]            == n_attacks * 4

    def test_augmented_labels_are_binary(self, dataset, X_normal):
        X, y = dataset
        _, y_aug, _ = augment_with_adversarial(X, y, X_normal)
        unique = np.unique(y_aug)
        assert set(unique).issubset({0, 1})


# ══════════════════════════════════════════════════════════════════════════════
#  3. TAUX D'ÉVASION — MODÈLE ORIGINAL
# ══════════════════════════════════════════════════════════════════════════════

class TestEvasionRateOriginalModel:

    def test_slow_and_low_high_evasion_rf(self, trained_rf):
        """Slow & Low doit avoir un taux d'évasion élevé sur le modèle original."""
        model, scaler = trained_rf
        X_slow = generate_slow_and_low(n_samples=100, seed=42)
        evasion = compute_evasion_rate(model, X_slow, scaler)
        # Taux d'évasion attendu élevé (>= 50%) sur le modèle non robuste
        assert evasion >= 0.5, f"Taux d'évasion Slow & Low attendu >= 50%, obtenu : {evasion:.1%}"

    def test_protocol_flip_low_evasion_rf(self, trained_rf, X_attacks):
        """Protocol Flip doit avoir un faible taux d'évasion (modèle robuste face à ça)."""
        model, scaler = trained_rf
        X_flip   = generate_protocol_flip(X_attacks)
        evasion  = compute_evasion_rate(model, X_flip, scaler)
        assert evasion <= 0.5, f"Taux d'évasion Protocol Flip attendu <= 50%, obtenu : {evasion:.1%}"

    def test_evasion_rate_in_range(self, trained_rf, X_attacks):
        """Le taux d'évasion est toujours dans [0, 1]."""
        model, scaler = trained_rf
        for generator in [
            generate_gaussian_noise(X_attacks),
            generate_protocol_flip(X_attacks),
            generate_slow_and_low(50),
        ]:
            rate = compute_evasion_rate(model, generator, scaler)
            assert 0.0 <= rate <= 1.0

    def test_evasion_with_isolation_forest(self, trained_if):
        """Slow & Low doit aussi tromper l'Isolation Forest."""
        model, scaler = trained_if
        X_slow  = generate_slow_and_low(n_samples=100, seed=42)
        evasion = compute_evasion_rate(model, X_slow, scaler)
        assert evasion >= 0.5, f"IF Slow & Low évasion >= 50% attendue, obtenu : {evasion:.1%}"


# ══════════════════════════════════════════════════════════════════════════════
#  4. RÉENTRAÎNEMENT ADVERSARIAL — MODÈLE ROBUSTE
# ══════════════════════════════════════════════════════════════════════════════

class TestAdversarialRetraining:

    def test_robust_model_is_random_forest(self, dataset, X_normal):
        X, y = dataset
        model_orig = RandomForestClassifier(n_estimators=50, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model_orig.fit(X_scaled, y)

        model_robust, stats = retrain_with_adversarial(model_orig, X_scaled, y, X_normal)
        assert isinstance(model_robust, RandomForestClassifier)

    def test_robust_model_trained_on_more_data(self, dataset, X_normal):
        """Le modèle robuste est entraîné sur plus de données."""
        X, y = dataset
        model_orig = RandomForestClassifier(n_estimators=50, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model_orig.fit(X_scaled, y)

        _, stats = retrain_with_adversarial(model_orig, X_scaled, y, X_normal)
        assert stats["total_samples"] > stats["original_samples"]

    def test_robust_model_better_against_slow_and_low(self, dataset, X_normal):
        """
        Le modèle robuste doit mieux détecter Slow & Low
        que le modèle original (taux d'évasion réduit).
        """
        X, y = dataset
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Modèle original
        model_orig = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model_orig.fit(X_scaled, y)

        # Modèle robuste
        model_robust, _ = retrain_with_adversarial(model_orig, X_scaled, y, X_normal)

        # Générer Slow & Low (jamais vu pendant l'entraînement du modèle original)
        X_slow = generate_slow_and_low(n_samples=200, seed=99)
        X_slow_scaled = scaler.transform(X_slow)

        evasion_orig   = compute_evasion_rate(model_orig,   X_slow_scaled)
        evasion_robust = compute_evasion_rate(model_robust, X_slow_scaled)

        assert evasion_robust <= evasion_orig, (
            f"Le modèle robuste devrait mieux détecter Slow & Low.\n"
            f"  Modèle original : {evasion_orig:.1%} d'évasion\n"
            f"  Modèle robuste  : {evasion_robust:.1%} d'évasion"
        )

    def test_robust_model_can_predict(self, dataset, X_normal):
        """Le modèle robuste peut faire des prédictions valides."""
        X, y = dataset
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model_orig = RandomForestClassifier(n_estimators=50, random_state=42)
        model_orig.fit(X_scaled, y)

        model_robust, _ = retrain_with_adversarial(model_orig, X_scaled, y, X_normal)
        preds = model_robust.predict(X_scaled)

        assert len(preds) == len(y)
        assert set(np.unique(preds)).issubset({0, 1})
