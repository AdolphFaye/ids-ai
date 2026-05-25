"""
feature_engineering/feature_extractor.py
───────────────────────────────────────────────────────────────
Module de Feature Engineering :
  1. Features statistiques (mean, std, min, max, median, count)
  2. Features comportementales agrégées (par IP, utilisateur, session)
  3. Fenêtrage temporel configurable
  4. Features dérivées réseau (packet_rate, byte_rate, IAT…)

Auteur : Alioune Badara Adolphe Faye
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger("FeatureExtractor")


class FeatureExtractor:
    """
    Extrait des features statistiques et comportementales depuis un DataFrame
    de flux réseau ou de logs.

    Colonnes attendues (CICIDS-style) :
        Flow Duration, Total Fwd Packets, Total Backward Packets,
        Total Length of Fwd Packets, Total Length of Bwd Packets,
        Flow IAT Mean, Flow IAT Std, Flow IAT Max, Flow IAT Min,
        Fwd IAT Mean, Bwd IAT Mean, Flow Bytes/s, Flow Packets/s,
        Source IP / Src IP, Destination IP / Dst IP, Protocol,
        Timestamp (optionnel)

    Exemple :
        fe = FeatureExtractor()
        df_features = fe.extract_all(df, label_col="Label")
    """

    # Noms canoniques des colonnes (mapping CICIDS → interne)
    COL_ALIASES: dict[str, list[str]] = {
        "src_ip":         ["Src IP", "Source IP", "src_ip", "SrcIP"],
        "dst_ip":         ["Dst IP", "Destination IP", "dst_ip", "DstIP"],
        "src_port":       ["Src Port", "src_port", "SrcPort"],
        "dst_port":       ["Dst Port", "dst_port", "DstPort"],
        "protocol":       ["Protocol", "protocol"],
        "timestamp":      ["Timestamp", "timestamp", "Flow ID"],
        "duration":       ["Flow Duration", "duration", "Duration"],
        "fwd_packets":    ["Total Fwd Packets", "fwd_packets"],
        "bwd_packets":    ["Total Backward Packets", "bwd_packets"],
        "fwd_bytes":      ["Total Length of Fwd Packets", "fwd_bytes"],
        "bwd_bytes":      ["Total Length of Bwd Packets", "bwd_bytes"],
        "flow_bytes_s":   ["Flow Bytes/s", "flow_bytes_s"],
        "flow_packets_s": ["Flow Packets/s", "flow_packets_s"],
        "iat_mean":       ["Flow IAT Mean", "iat_mean"],
        "iat_std":        ["Flow IAT Std", "iat_std"],
        "iat_max":        ["Flow IAT Max", "iat_max"],
        "iat_min":        ["Flow IAT Min", "iat_min"],
    }

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        fe_cfg = self.config["feature_engineering"]

        self._window_seconds   = fe_cfg["time_window_seconds"]
        self._aggregate_by     = fe_cfg["aggregate_by"]
        self._stat_functions   = fe_cfg["statistical_features"]
        self._behavioral_feats = fe_cfg["behavioral_features"]

        logger.info(
            f"FeatureExtractor initialisé — "
            f"fenêtre={self._window_seconds}s | "
            f"agrégation par {self._aggregate_by}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  API PUBLIQUE
    # ─────────────────────────────────────────────────────────────────────────

    def extract_all(
        self,
        df: pd.DataFrame,
        label_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Pipeline complet d'extraction de features.

        Args:
            df        : DataFrame prétraité.
            label_col : Colonne label à préserver (non transformée).

        Returns:
            DataFrame enrichi de nouvelles features.
        """
        logger.info(f"Extraction de features — {len(df):,} lignes")
        df = df.copy()

        # Normaliser les noms de colonnes
        df = self._normalize_column_names(df)

        labels = None
        if label_col and label_col in df.columns:
            labels = df[label_col].copy()
            df.drop(columns=[label_col], inplace=True)

        # ── 1. Features dérivées directes ──────────────────────────────────
        df = self._add_derived_features(df)

        # ── 2. Features comportementales agrégées ──────────────────────────
        df = self._add_behavioral_features(df)

        # ── 3. Features de fenêtrage temporel ──────────────────────────────
        if "timestamp" in df.columns:
            df = self._add_time_window_features(df)

        # ── 4. Supprimer les NaN introduits par les rolling windows ─────────
        df = df.fillna(0)

        # Remettre le label
        if labels is not None:
            df[label_col] = labels.values

        n_new = df.shape[1] - (1 if label_col else 0)
        logger.success(
            f"Extraction terminée — {df.shape[1]} features au total "
            f"(dont {n_new} features engineered)"
        )
        return df

    # ─────────────────────────────────────────────────────────────────────────
    #  FEATURES DÉRIVÉES DIRECTES
    # ─────────────────────────────────────────────────────────────────────────

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule des ratios et features réseau directement depuis les colonnes brutes.
        """
        feats_added = []

        # Ratio bytes entrants / sortants
        if "fwd_bytes" in df.columns and "bwd_bytes" in df.columns:
            total = df["fwd_bytes"] + df["bwd_bytes"]
            df["bytes_ratio_fwd_bwd"] = np.where(
                total > 0, df["fwd_bytes"] / (total + 1e-9), 0
            )
            df["total_bytes"] = total
            feats_added += ["bytes_ratio_fwd_bwd", "total_bytes"]

        # Ratio paquets entrants / sortants
        if "fwd_packets" in df.columns and "bwd_packets" in df.columns:
            total_pkts = df["fwd_packets"] + df["bwd_packets"]
            df["packet_ratio_fwd_bwd"] = np.where(
                total_pkts > 0, df["fwd_packets"] / (total_pkts + 1e-9), 0
            )
            df["total_packets"] = total_pkts
            feats_added += ["packet_ratio_fwd_bwd", "total_packets"]

        # Taux de bytes/s si non présent
        if "flow_bytes_s" not in df.columns:
            if "total_bytes" in df.columns and "duration" in df.columns:
                df["flow_bytes_s"] = np.where(
                    df["duration"] > 0,
                    df["total_bytes"] / (df["duration"] / 1e6 + 1e-9),
                    0,
                )
                feats_added.append("flow_bytes_s")

        # Taux de paquets/s si non présent
        if "flow_packets_s" not in df.columns:
            if "total_packets" in df.columns and "duration" in df.columns:
                df["flow_packets_s"] = np.where(
                    df["duration"] > 0,
                    df["total_packets"] / (df["duration"] / 1e6 + 1e-9),
                    0,
                )
                feats_added.append("flow_packets_s")

        # Variance de l'IAT
        if "iat_std" in df.columns:
            df["iat_variance"] = df["iat_std"] ** 2
            feats_added.append("iat_variance")

        # Amplitude IAT (max - min)
        if "iat_max" in df.columns and "iat_min" in df.columns:
            df["iat_range"] = df["iat_max"] - df["iat_min"]
            feats_added.append("iat_range")

        if feats_added:
            logger.info(f"Features dérivées ajoutées : {feats_added}")
        return df

    # ─────────────────────────────────────────────────────────────────────────
    #  FEATURES COMPORTEMENTALES AGRÉGÉES
    # ─────────────────────────────────────────────────────────────────────────

    def _add_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule des profils comportementaux agrégés par :
        - IP source
        - IP destination
        - Protocol

        Pour chaque groupe, calcule des statistiques sur les colonnes
        numériques principales.
        """
        num_cols_for_agg = [
            c for c in ["flow_bytes_s", "flow_packets_s", "duration",
                        "total_bytes", "total_packets", "iat_mean"]
            if c in df.columns
        ]

        if not num_cols_for_agg:
            logger.warning("Aucune colonne numérique disponible pour l'agrégation comportementale.")
            return df

        agg_groups = {
            "src_ip":  "srcip",
            "dst_ip":  "dstip",
            "protocol": "proto",
        }

        for col_name, suffix in agg_groups.items():
            if col_name not in df.columns:
                continue

            for num_col in num_cols_for_agg:
                for stat in ["mean", "std", "count"]:
                    feat_name = f"{suffix}_{num_col}_{stat}"
                    agg = df.groupby(col_name)[num_col].transform(stat)
                    df[feat_name] = agg.fillna(0)

        logger.info("Features comportementales agrégées ajoutées (src_ip, dst_ip, protocol)")
        return df

    # ─────────────────────────────────────────────────────────────────────────
    #  FEATURES TEMPORELLES (FENÊTRAGE)
    # ─────────────────────────────────────────────────────────────────────────

    def _add_time_window_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule des features sur une fenêtre glissante temporelle.
        Nécessite une colonne 'timestamp' numérique (epoch secondes).

        Features calculées sur la fenêtre :
            - connections_in_window  : nombre de flux dans la fenêtre
            - bytes_in_window        : total de bytes dans la fenêtre
            - unique_src_ips_window  : IPs sources uniques
            - unique_dst_ips_window  : IPs destinations uniques
        """
        W = self._window_seconds

        try:
            ts = pd.to_numeric(df["timestamp"], errors="coerce")
            df["timestamp_num"] = ts
        except Exception:
            logger.warning("Impossible de convertir le timestamp — fenêtrage ignoré")
            return df

        df = df.sort_values("timestamp_num").reset_index(drop=True)

        window_features = {
            "connections_in_window":  [],
            "bytes_in_window":        [],
            "unique_src_ips_window":  [],
            "unique_dst_ips_window":  [],
        }

        has_bytes  = "total_bytes" in df.columns
        has_src_ip = "src_ip" in df.columns
        has_dst_ip = "dst_ip" in df.columns

        timestamps = df["timestamp_num"].values

        for i, t in enumerate(timestamps):
            mask = (timestamps >= t - W) & (timestamps <= t)
            window_df = df.iloc[mask]

            window_features["connections_in_window"].append(len(window_df))
            window_features["bytes_in_window"].append(
                window_df["total_bytes"].sum() if has_bytes else 0
            )
            window_features["unique_src_ips_window"].append(
                window_df["src_ip"].nunique() if has_src_ip else 0
            )
            window_features["unique_dst_ips_window"].append(
                window_df["dst_ip"].nunique() if has_dst_ip else 0
            )

        for feat, values in window_features.items():
            df[feat] = values

        df.drop(columns=["timestamp_num"], inplace=True, errors="ignore")
        logger.info(f"Features de fenêtrage temporel ajoutées (fenêtre={W}s)")
        return df

    # ─────────────────────────────────────────────────────────────────────────
    #  UTILITAIRES
    # ─────────────────────────────────────────────────────────────────────────

    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Renomme les colonnes du DataFrame vers les noms canoniques internes
        en utilisant la table COL_ALIASES.
        """
        rename_map = {}
        for canonical, aliases in self.COL_ALIASES.items():
            for alias in aliases:
                if alias in df.columns and canonical not in df.columns:
                    rename_map[alias] = canonical
                    break
        if rename_map:
            df = df.rename(columns=rename_map)
            logger.info(f"Colonnes renommées : {rename_map}")
        return df

    def get_feature_names(self, df: pd.DataFrame) -> list[str]:
        """Retourne la liste des features (hors label)."""
        return [c for c in df.columns if not c.lower() in ("label", "class", "attack")]
