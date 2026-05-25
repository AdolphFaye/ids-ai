"""
pipeline.py
───────────────────────────────────────────────────────────────
Point d'entrée principal du pipeline Data Engineering.
Orchestre : chargement → prétraitement → feature engineering
            → rééquilibrage → split → export

Auteur : Alioune Badara Adolphe Faye

Usage :
    python pipeline.py --dataset CICIDS2017 \
                       --input  data/raw/cicids2017.csv \
                       --name   cicids2017
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from src.ingestion.data_loader           import DataLoader
from src.preprocessing.data_preprocessor import DataPreprocessor
from src.preprocessing.class_balancer    import ClassBalancer
from src.preprocessing.data_splitter     import DataSplitter
from src.feature_engineering.feature_extractor import FeatureExtractor
from src.utils.logger                    import get_logger
from src.utils.config_loader             import load_config

logger = get_logger("Pipeline")


def run_pipeline(
    dataset_type: str,
    input_path:   str,
    dataset_name: str,
    config_path:  str = "config.yaml",
    nrows:        int | None = None,
    skip_smote:   bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Pipeline complet Data Engineering.

    Args:
        dataset_type  : "CICIDS2017" | "CICIDS2018" | "NSL-KDD"
        input_path    : Chemin du fichier CSV brut.
        dataset_name  : Nom/préfixe pour les fichiers de sortie.
        config_path   : Chemin vers config.yaml.
        nrows         : Limiter le nombre de lignes (pour tests rapides).
        skip_smote    : Sauter le SMOTE si déjà équilibré.

    Returns:
        dict {"train": df_train, "val": df_val, "test": df_test}
    """
    t0 = time.time()
    config = load_config(config_path)
    label_col = config["datasets"]["label_column"][dataset_type]

    logger.info("=" * 60)
    logger.info(f"  PIPELINE IDS — Data Engineering")
    logger.info(f"  Dataset      : {dataset_type}")
    logger.info(f"  Fichier      : {input_path}")
    logger.info(f"  Label col    : {label_col}")
    logger.info("=" * 60)

    # ── ÉTAPE 1 : Chargement ─────────────────────────────────────────────────
    logger.info("[1/5] Chargement des données...")
    loader = DataLoader(dataset_type=dataset_type, config_path=config_path)
    df_raw = loader.load_csv(input_path, nrows=nrows)

    # ── ÉTAPE 2 : Prétraitement ───────────────────────────────────────────────
    logger.info("[2/5] Prétraitement...")
    preprocessor = DataPreprocessor(dataset_type=dataset_type, config_path=config_path)
    df_clean = preprocessor.fit_transform(df_raw, label_col=label_col)

    summary = preprocessor.summary(df_clean)
    logger.info(f"Résumé : {summary}")

    # ── ÉTAPE 3 : Feature Engineering ────────────────────────────────────────
    logger.info("[3/5] Feature Engineering...")
    extractor = FeatureExtractor(config_path=config_path)
    df_features = extractor.extract_all(df_clean, label_col=label_col)

    # ── ÉTAPE 4 : Rééquilibrage des classes ──────────────────────────────────
    logger.info("[4/5] Rééquilibrage des classes...")

    if not skip_smote:
        X = df_features.drop(columns=[label_col])
        y = df_features[label_col]

        balancer = ClassBalancer(config_path=config_path)
        X_bal, y_bal = balancer.fit_resample(X, y)
        df_balanced = X_bal.copy()
        df_balanced[label_col] = y_bal.values
    else:
        df_balanced = df_features
        logger.info("SMOTE ignoré (--skip-smote)")

    # ── ÉTAPE 5 : Split & Export ──────────────────────────────────────────────
    logger.info("[5/5] Découpage train/val/test et export...")
    splitter = DataSplitter(config_path=config_path)
    splits = splitter.split_and_save(
        df_balanced,
        label_col=label_col,
        dataset_name=dataset_name,
        raw_path=input_path,   # Suppression du fichier brut (sécurité)
    )

    elapsed = time.time() - t0
    logger.success(f"Pipeline terminé en {elapsed:.1f}s")
    logger.info(f"Fichiers exportés dans : {config['paths']['processed_data']}")

    return splits


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="IDS Data Engineering Pipeline — Alioune Badara Adolphe Faye"
    )
    parser.add_argument(
        "--dataset", "-d",
        choices=["CICIDS2017", "CICIDS2018", "NSL-KDD"],
        required=True,
        help="Type de dataset à traiter",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Chemin vers le fichier CSV brut",
    )
    parser.add_argument(
        "--name", "-n",
        default="dataset",
        help="Nom/préfixe pour les fichiers de sortie (défaut: dataset)",
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Chemin vers le fichier de configuration (défaut: config.yaml)",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Limiter le nombre de lignes chargées (pour tests rapides)",
    )
    parser.add_argument(
        "--skip-smote",
        action="store_true",
        help="Désactiver le rééquilibrage SMOTE",
    )

    args = parser.parse_args()

    try:
        run_pipeline(
            dataset_type=args.dataset,
            input_path=args.input,
            dataset_name=args.name,
            config_path=args.config,
            nrows=args.nrows,
            skip_smote=args.skip_smote,
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Erreur inattendue : {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
