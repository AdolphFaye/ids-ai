"""
preprocessing/class_balancer.py
───────────────────────────────────────────────────────────────
Gestion du déséquilibre des classes (attaques rares vs trafic normal).
Stratégie : SMOTE (Synthetic Minority Oversampling TEchnique)

Auteur : Alioune Badara Adolphe Faye
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from collections import Counter

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger("ClassBalancer")


class ClassBalancer:
    """
    Rééquilibre les classes dans un dataset fortement déséquilibré.

    Méthodes disponibles :
        - smote        : suréchantillonnage synthétique de la classe minoritaire
        - class_weight : retourne un dictionnaire de poids (pour sklearn)
        - none         : pas de rééquilibrage

    Exemple :
        balancer = ClassBalancer(strategy="smote")
        X_bal, y_bal = balancer.fit_resample(X_train, y_train)
        weights = balancer.compute_class_weights(y_train)
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config  = load_config(config_path)
        ci_cfg       = self.config["class_imbalance"]
        self.strategy        = ci_cfg["strategy"]
        self.sampling_ratio  = ci_cfg["sampling_ratio"]
        self._smote = None
        logger.info(f"ClassBalancer initialisé — stratégie : {self.strategy}")

    # ─────────────────────────────────────────────────────────────────────────

    def fit_resample(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Applique le rééquilibrage sur (X, y).

        Args:
            X : Features (sans label).
            y : Vecteur de labels.

        Returns:
            (X_resampled, y_resampled)
        """
        if self.strategy == "none":
            logger.info("Aucun rééquilibrage appliqué.")
            return X, y

        if self.strategy == "smote":
            return self._apply_smote(X, y)

        if self.strategy == "class_weight":
            logger.info("Stratégie class_weight — pas de rééchantillonnage, utilisez compute_class_weights()")
            return X, y

        raise ValueError(f"Stratégie inconnue : {self.strategy}")

    # ─────────────────────────────────────────────────────────────────────────

    def _apply_smote(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Applique SMOTE pour suréchantillonner les classes minoritaires."""
        dist_before = Counter(y)
        logger.info(f"Distribution AVANT SMOTE : {dict(dist_before)}")

        # Calculer le sampling_strategy adaptatif
        majority_count = max(dist_before.values())
        target_count   = int(majority_count * self.sampling_ratio)

        sampling_strategy = {
            cls: max(count, target_count)
            for cls, count in dist_before.items()
            if count < majority_count
        }

        if not sampling_strategy:
            logger.warning("Les classes sont déjà équilibrées — SMOTE non appliqué.")
            return X, y

        self._smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=self.config["output"]["random_state"],
            n_jobs=-1,
        )

        cols = X.columns.tolist()
        X_arr, y_arr = self._smote.fit_resample(X.values, y.values)

        X_res = pd.DataFrame(X_arr, columns=cols)
        y_res = pd.Series(y_arr, name=y.name)

        dist_after = Counter(y_res)
        logger.success(f"Distribution APRÈS SMOTE : {dict(dist_after)}")
        logger.info(
            f"Lignes ajoutées : {len(X_res) - len(X):,} "
            f"({len(X):,} → {len(X_res):,})"
        )
        return X_res, y_res

    # ─────────────────────────────────────────────────────────────────────────

    def compute_class_weights(self, y: pd.Series) -> dict:
        """
        Calcule des poids inversement proportionnels à la fréquence de chaque classe.
        À passer directement à class_weight dans sklearn.

        Returns:
            dict {classe: poids}
        """
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        weight_dict = dict(zip(classes, weights))
        logger.info(f"Poids des classes calculés : {weight_dict}")
        return weight_dict
