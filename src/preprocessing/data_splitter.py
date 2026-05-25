"""
preprocessing/data_splitter.py
───────────────────────────────────────────────────────────────
Découpage train / val / test et export sécurisé (parquet ou CSV).
Les données brutes ne sont JAMAIS persistées en clair (exigence sécurité).

Auteur : Alioune Badara Adolphe Faye
"""

from __future__ import annotations

import os
import shutil
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger("DataSplitter")


class DataSplitter:
    """
    Découpe un DataFrame en ensembles train / val / test
    et les exporte dans data/processed/.

    Respecte l'exigence de sécurité :
        Les données brutes ne sont pas persistées en clair sur le disque.

    Exemple :
        splitter = DataSplitter()
        splits = splitter.split_and_save(df, label_col="Label")
        # splits = {"train": df_train, "val": df_val, "test": df_test}
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config   = load_config(config_path)
        out_cfg       = self.config["output"]
        paths_cfg     = self.config["paths"]

        self.train_ratio   = out_cfg["train_ratio"]
        self.val_ratio     = out_cfg["val_ratio"]
        self.test_ratio    = out_cfg["test_ratio"]
        self.random_state  = out_cfg["random_state"]
        self.fmt           = out_cfg["format"]           # parquet | csv
        self.processed_dir = Path(paths_cfg["processed_data"])
        self.clear_raw     = self.config["security"]["clear_raw_after_processing"]

        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6, \
            "Les ratios train/val/test doivent sommer à 1.0"

        logger.info(
            f"DataSplitter — "
            f"train={self.train_ratio} | val={self.val_ratio} | test={self.test_ratio} | "
            f"format={self.fmt}"
        )

    # ─────────────────────────────────────────────────────────────────────────

    def split_and_save(
        self,
        df: pd.DataFrame,
        label_col: str,
        dataset_name: str = "dataset",
        raw_path: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Stratified split + export.

        Args:
            df           : DataFrame prétraité avec features + label.
            label_col    : Nom de la colonne label.
            dataset_name : Préfixe des fichiers de sortie.
            raw_path     : Chemin du fichier brut à supprimer après traitement
                           (respect de la politique sécurité).

        Returns:
            {"train": df_train, "val": df_val, "test": df_test}
        """
        logger.info(f"Découpage stratifié — {len(df):,} lignes")

        X = df.drop(columns=[label_col])
        y = df[label_col]

        # 1. Train vs (val + test)
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y,
            test_size=self.val_ratio + self.test_ratio,
            stratify=y,
            random_state=self.random_state,
        )

        # 2. Val vs test
        val_ratio_adjusted = self.val_ratio / (self.val_ratio + self.test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp,
            test_size=1 - val_ratio_adjusted,
            stratify=y_tmp,
            random_state=self.random_state,
        )

        # Reconstituer les DataFrames complets
        splits = {
            "train": X_train.assign(**{label_col: y_train.values}),
            "val":   X_val.assign(**{label_col: y_val.values}),
            "test":  X_test.assign(**{label_col: y_test.values}),
        }

        for split_name, split_df in splits.items():
            logger.info(
                f"  {split_name:<6}: {len(split_df):>7,} lignes | "
                f"distribution : {split_df[label_col].value_counts().to_dict()}"
            )

        # Export
        self._export(splits, dataset_name)

        # Sécurité : supprimer le fichier brut si demandé
        if raw_path and self.clear_raw:
            self._secure_delete(raw_path)

        logger.success("Découpage et export terminés.")
        return splits

    # ─────────────────────────────────────────────────────────────────────────

    def _export(self, splits: dict[str, pd.DataFrame], dataset_name: str) -> None:
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_df in splits.items():
            if self.fmt == "parquet":
                fpath = self.processed_dir / f"{dataset_name}_{split_name}.parquet"
                split_df.to_parquet(fpath, index=False)
            else:
                fpath = self.processed_dir / f"{dataset_name}_{split_name}.csv"
                split_df.to_csv(fpath, index=False)

            size_mb = fpath.stat().st_size / 1e6
            logger.info(f"Exporté : {fpath.name} ({size_mb:.1f} MB)")

    def _secure_delete(self, path: str) -> None:
        """Supprime le fichier brut du disque (ne pas persister en clair)."""
        p = Path(path)
        if p.exists():
            p.unlink()
            logger.info(f"Fichier brut supprimé (sécurité) : {p.name}")
        else:
            logger.warning(f"Fichier brut introuvable pour suppression : {path}")

    # ─────────────────────────────────────────────────────────────────────────

    def load_split(self, dataset_name: str, split: str) -> pd.DataFrame:
        """
        Recharge un split depuis le dossier processed.

        Args:
            dataset_name : Préfixe du fichier (ex: "cicids2017").
            split        : "train" | "val" | "test"
        """
        if self.fmt == "parquet":
            fpath = self.processed_dir / f"{dataset_name}_{split}.parquet"
            return pd.read_parquet(fpath)
        else:
            fpath = self.processed_dir / f"{dataset_name}_{split}.csv"
            return pd.read_csv(fpath)
