"""
ingestion/data_loader.py
───────────────────────────────────────────────────────────────
Module d'ingestion des données :
  - Fichiers CSV (CICIDS 2017/2018, NSL-KDD)
  - Fichiers PCAP (captures réseau)

Auteur : Alioune Badara Adolphe Faye
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger("DataLoader")


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTES — Colonnes connues des datasets publics
# ─────────────────────────────────────────────────────────────────────────────

NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "class", "difficulty_level",
]


# ─────────────────────────────────────────────────────────────────────────────
#  CLASSE PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────

class DataLoader:
    """
    Charge les données depuis des fichiers CSV ou PCAP.

    Supporte :
        - CICIDS 2017 / 2018 (CSV)
        - NSL-KDD            (CSV sans header)
        - Fichiers PCAP      (via Scapy)

    Exemple d'utilisation :
        loader = DataLoader(dataset_type="CICIDS2017")
        df = loader.load_csv("data/raw/cicids2017.csv")
    """

    SUPPORTED_DATASETS = ("CICIDS2017", "CICIDS2018", "NSL-KDD")

    def __init__(self, dataset_type: str = "CICIDS2017", config_path: str = "config.yaml"):
        if dataset_type not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Dataset non supporté : '{dataset_type}'. "
                f"Valeurs acceptées : {self.SUPPORTED_DATASETS}"
            )
        self.dataset_type = dataset_type
        self.config = load_config(config_path)
        self._label_col = self.config["datasets"]["label_column"][dataset_type]
        self._normal_label = self.config["datasets"]["normal_label"][dataset_type]
        logger.info(f"DataLoader initialisé — dataset : {dataset_type}")

    # ── CSV ──────────────────────────────────────────────────────────────────

    def load_csv(
        self,
        filepath: str,
        nrows: Optional[int] = None,
        encoding: str = "utf-8",
    ) -> pd.DataFrame:
        """
        Charge un fichier CSV.

        Args:
            filepath : Chemin vers le fichier CSV.
            nrows    : Nombre de lignes à charger (None = tout).
            encoding : Encodage du fichier.

        Returns:
            DataFrame pandas brut.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Fichier CSV introuvable : {filepath}")

        logger.info(f"Chargement CSV : {path.name} ({self.dataset_type})")

        # NSL-KDD n'a pas de header
        if self.dataset_type == "NSL-KDD":
            df = pd.read_csv(
                path,
                header=None,
                names=NSL_KDD_COLUMNS,
                nrows=nrows,
                encoding=encoding,
            )
            # Supprimer la colonne de difficulté inutile pour le ML
            if "difficulty_level" in df.columns:
                df.drop(columns=["difficulty_level"], inplace=True)
        else:
            df = pd.read_csv(
                path,
                nrows=nrows,
                encoding=encoding,
                low_memory=False,
            )
            # Nettoyer les espaces dans les noms de colonnes (CICIDS)
            df.columns = df.columns.str.strip()

        logger.success(
            f"CSV chargé — {len(df):,} lignes | {df.shape[1]} colonnes"
        )
        self._log_label_distribution(df)
        return df

    def load_multiple_csv(self, folder: str, pattern: str = "*.csv") -> pd.DataFrame:
        """
        Charge et concatène tous les fichiers CSV d'un dossier.

        Args:
            folder  : Chemin du dossier contenant les CSV.
            pattern : Motif glob (défaut : *.csv).

        Returns:
            DataFrame concaténé.
        """
        files = list(Path(folder).glob(pattern))
        if not files:
            raise FileNotFoundError(f"Aucun fichier CSV trouvé dans : {folder}")

        logger.info(f"Chargement de {len(files)} fichier(s) CSV depuis {folder}")
        frames = []
        for f in sorted(files):
            frames.append(self.load_csv(str(f)))

        df = pd.concat(frames, ignore_index=True)
        logger.success(f"Concaténation terminée — {len(df):,} lignes au total")
        return df

    # ── PCAP ─────────────────────────────────────────────────────────────────

    def load_pcap(self, filepath: str, max_packets: int = 100_000) -> pd.DataFrame:
        """
        Charge un fichier PCAP et extrait les features réseau de base.

        Chaque ligne du DataFrame correspond à un paquet réseau avec :
        timestamp, src_ip, dst_ip, src_port, dst_port, protocol,
        packet_length, ttl.

        Args:
            filepath    : Chemin vers le fichier .pcap / .pcapng.
            max_packets : Limite du nombre de paquets à analyser.

        Returns:
            DataFrame des paquets.
        """
        try:
            from scapy.all import rdpcap, IP, TCP, UDP, ICMP
        except ImportError:
            raise ImportError(
                "Scapy est requis pour lire les fichiers PCAP.\n"
                "Installez-le avec : pip install scapy"
            )

        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Fichier PCAP introuvable : {filepath}")

        logger.info(f"Lecture PCAP : {path.name} (max {max_packets:,} paquets)")

        packets = rdpcap(str(path))
        records = []

        for i, pkt in enumerate(packets):
            if i >= max_packets:
                break
            if not pkt.haslayer(IP):
                continue

            ip = pkt[IP]
            record = {
                "timestamp":      float(pkt.time),
                "src_ip":         ip.src,
                "dst_ip":         ip.dst,
                "protocol":       ip.proto,
                "packet_length":  len(pkt),
                "ttl":            ip.ttl,
                "src_port":       None,
                "dst_port":       None,
                "tcp_flags":      None,
            }

            if pkt.haslayer(TCP):
                tcp = pkt[TCP]
                record["src_port"]  = tcp.sport
                record["dst_port"]  = tcp.dport
                record["tcp_flags"] = str(tcp.flags)
            elif pkt.haslayer(UDP):
                udp = pkt[UDP]
                record["src_port"] = udp.sport
                record["dst_port"] = udp.dport

            records.append(record)

        df = pd.DataFrame(records)
        logger.success(f"PCAP chargé — {len(df):,} paquets IP extraits")
        return df

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log_label_distribution(self, df: pd.DataFrame) -> None:
        """Affiche la distribution des labels dans les logs."""
        if self._label_col not in df.columns:
            return
        dist = df[self._label_col].value_counts()
        logger.info("Distribution des labels :")
        for label, count in dist.items():
            pct = count / len(df) * 100
            logger.info(f"  {str(label):<30} {count:>8,}  ({pct:.1f}%)")

    @property
    def label_column(self) -> str:
        return self._label_col

    @property
    def normal_label(self) -> str:
        return self._normal_label
