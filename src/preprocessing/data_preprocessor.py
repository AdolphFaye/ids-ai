"""
preprocessing/data_preprocessor.py
───────────────────────────────────────────────────────────────
Module de prétraitement des données :
  1. Suppression des doublons et valeurs infinies
  2. Gestion des valeurs manquantes
  3. Normalisation des features numériques
  4. Encodage des variables catégorielles

Auteur : Alioune Badara Adolphe Faye
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
)
from sklearn.impute import SimpleImputer

from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger("Preprocessor")


class DataPreprocessor:
    """
    Prétraitement complet d'un DataFrame pour un IDS.

    Étapes :
        1. drop_duplicates()         — suppression des doublons
        2. remove_infinite()         — remplacement inf/-inf par NaN
        3. handle_missing_values()   — imputation
        4. normalize()               — mise à l'échelle numérique
        5. encode_categoricals()     — encodage des colonnes textuelles

    Exemple :
        prep = DataPreprocessor(dataset_type="CICIDS2017")
        df_clean = prep.fit_transform(df, label_col="Label")
        df_test_clean = prep.transform(df_test)
    """

    SCALERS = {
        "standard": StandardScaler,
        "minmax":   MinMaxScaler,
        "robust":   RobustScaler,
    }

    def __init__(
        self,
        dataset_type: str = "CICIDS2017",
        config_path: str = "config.yaml",
    ):
        self.dataset_type = dataset_type
        self.config = load_config(config_path)
        pp_cfg = self.config["preprocessing"]

        # Paramètres extraits du config
        self._missing_strategy    = pp_cfg["missing_values"]["strategy"]
        self._drop_col_threshold  = pp_cfg["missing_values"]["threshold_drop_col"]
        self._norm_method         = pp_cfg["normalization"]["method"]
        self._enc_method          = pp_cfg["encoding"]["method"]
        self._remove_duplicates   = pp_cfg["remove_duplicates"]
        self._remove_infinite     = pp_cfg["remove_infinite"]

        # Objets sklearn (remplis après fit)
        self._scaler: Optional[StandardScaler | MinMaxScaler | RobustScaler] = None
        self._imputer: Optional[SimpleImputer] = None
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._numeric_cols: list[str] = []
        self._cat_cols: list[str] = []
        self._fitted = False

        logger.info(
            f"Preprocessor initialisé — "
            f"normalisation={self._norm_method} | "
            f"imputation={self._missing_strategy} | "
            f"encodage={self._enc_method}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  API PUBLIQUE
    # ─────────────────────────────────────────────────────────────────────────

    def fit_transform(
        self,
        df: pd.DataFrame,
        label_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Ajuste le préprocesseur sur df et retourne le DataFrame transformé.

        Args:
            df        : DataFrame brut.
            label_col : Colonne label à exclure du prétraitement.

        Returns:
            DataFrame prétraité.
        """
        logger.info(f"fit_transform — {len(df):,} lignes | {df.shape[1]} colonnes")
        df = df.copy()

        # Séparer le label temporairement
        labels = None
        if label_col and label_col in df.columns:
            labels = df[label_col].copy()
            df.drop(columns=[label_col], inplace=True)

        df = self._drop_duplicates(df)
        df = self._remove_inf(df)
        df = self._drop_high_missing_columns(df)
        df = self._identify_column_types(df)
        df = self._fit_impute(df)
        df = self._fit_scale(df)
        df = self._fit_encode(df)

        self._fitted = True

        # Remettre le label
        if labels is not None:
            # Ré-aligner l'index après drop_duplicates
            labels = labels.loc[df.index]
            df[label_col] = labels.values

        logger.success(
            f"fit_transform terminé — {len(df):,} lignes | {df.shape[1]} colonnes"
        )
        return df

    def transform(self, df: pd.DataFrame, label_col: Optional[str] = None) -> pd.DataFrame:
        """
        Applique le préprocesseur déjà ajusté à un nouveau DataFrame (test/val).

        Args:
            df        : DataFrame brut.
            label_col : Colonne label à exclure du prétraitement.

        Returns:
            DataFrame prétraité.
        """
        if not self._fitted:
            raise RuntimeError("Le préprocesseur n'a pas encore été ajusté. Appelez fit_transform() d'abord.")

        logger.info(f"transform — {len(df):,} lignes")
        df = df.copy()

        labels = None
        if label_col and label_col in df.columns:
            labels = df[label_col].copy()
            df.drop(columns=[label_col], inplace=True)

        df = self._remove_inf(df)
        df = self._apply_impute(df)
        df = self._apply_scale(df)
        df = self._apply_encode(df)

        if labels is not None:
            df[label_col] = labels.values

        return df

    # ─────────────────────────────────────────────────────────────────────────
    #  ÉTAPES INTERNES
    # ─────────────────────────────────────────────────────────────────────────

    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._remove_duplicates:
            return df
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        if removed > 0:
            logger.info(f"Doublons supprimés : {removed:,}")
        return df.reset_index(drop=True)

    def _remove_inf(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._remove_infinite:
            return df
        num_cols = df.select_dtypes(include=[np.number]).columns
        mask = np.isinf(df[num_cols])
        count = mask.values.sum()
        if count > 0:
            df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
            logger.info(f"Valeurs infinies remplacées par NaN : {count:,}")
        return df

    def _drop_high_missing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        thresh = self._drop_col_threshold
        missing_ratio = df.isnull().mean()
        cols_to_drop = missing_ratio[missing_ratio > thresh].index.tolist()
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            logger.warning(
                f"Colonnes supprimées (>{thresh*100:.0f}% NaN) : {cols_to_drop}"
            )
        return df

    def _identify_column_types(self, df: pd.DataFrame) -> pd.DataFrame:
        self._numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self._cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        logger.info(
            f"Colonnes numériques : {len(self._numeric_cols)} | "
            f"Catégorielles : {len(self._cat_cols)}"
        )
        return df

    # ── Imputation ────────────────────────────────────────────────────────────

    def _fit_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._numeric_cols:
            return df
        strategy = self._missing_strategy
        if strategy == "drop":
            before = len(df)
            df = df.dropna(subset=self._numeric_cols)
            logger.info(f"Lignes supprimées (NaN) : {before - len(df):,}")
            return df.reset_index(drop=True)
        if strategy == "fill_zero":
            df[self._numeric_cols] = df[self._numeric_cols].fillna(0)
            return df

        sk_strategy = "mean" if strategy == "mean" else "median"
        self._imputer = SimpleImputer(strategy=sk_strategy)
        df[self._numeric_cols] = self._imputer.fit_transform(df[self._numeric_cols])
        missing_total = df[self._numeric_cols].isnull().sum().sum()
        logger.info(f"Imputation ({strategy}) appliquée — NaN restants : {missing_total}")
        return df

    def _apply_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._imputer is None or not self._numeric_cols:
            return df
        cols = [c for c in self._numeric_cols if c in df.columns]
        df[cols] = self._imputer.transform(df[cols])
        return df

    # ── Normalisation ─────────────────────────────────────────────────────────

    def _fit_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._numeric_cols:
            return df
        scaler_cls = self.SCALERS.get(self._norm_method, StandardScaler)
        self._scaler = scaler_cls()
        df[self._numeric_cols] = self._scaler.fit_transform(df[self._numeric_cols])
        logger.info(f"Normalisation ({self._norm_method}) appliquée")
        return df

    def _apply_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._scaler is None or not self._numeric_cols:
            return df
        cols = [c for c in self._numeric_cols if c in df.columns]
        df[cols] = self._scaler.transform(df[cols])
        return df

    # ── Encodage catégoriel ───────────────────────────────────────────────────

    def _fit_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._cat_cols:
            return df
        for col in self._cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self._label_encoders[col] = le
        logger.info(f"Encodage ({self._enc_method}) appliqué sur {len(self._cat_cols)} colonnes")
        return df

    def _apply_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, le in self._label_encoders.items():
            if col not in df.columns:
                continue
            # Gérer les classes inconnues (unseen labels)
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in known else le.classes_[0]
            )
            df[col] = le.transform(df[col])
        return df

    # ─────────────────────────────────────────────────────────────────────────
    #  RAPPORT
    # ─────────────────────────────────────────────────────────────────────────

    def summary(self, df: pd.DataFrame) -> dict:
        """Retourne un résumé des statistiques du DataFrame prétraité."""
        return {
            "rows":          len(df),
            "columns":       df.shape[1],
            "numeric_cols":  len(self._numeric_cols),
            "cat_cols":      len(self._cat_cols),
            "missing_total": int(df.isnull().sum().sum()),
            "memory_mb":     round(df.memory_usage(deep=True).sum() / 1e6, 2),
        }
