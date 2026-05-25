"""
tests/test_data_loader.py
───────────────────────────────────────────────────────────────
Tests unitaires — Module DataLoader
Auteur : Alioune Badara Adolphe Faye
"""

import pytest
import pandas as pd
import numpy as np
from io import StringIO
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.ingestion.data_loader import DataLoader, NSL_KDD_COLUMNS


# ─────────────────────────────────────────────────────────────────────────────
#  FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_cicids_csv(tmp_path) -> str:
    """Crée un petit CSV CICIDS2017 factice."""
    data = (
        "Dst Port,Protocol,Flow Duration,Total Fwd Packets,"
        "Total Backward Packets,Total Length of Fwd Packets,"
        "Total Length of Bwd Packets,Flow Bytes/s,Flow Packets/s,"
        "Flow IAT Mean,Flow IAT Std,Flow IAT Max,Flow IAT Min,"
        "Src IP,Dst IP,Label\n"
        "80,6,100,10,5,500,250,5000,150,10,2,50,1,192.168.1.1,10.0.0.1,BENIGN\n"
        "443,6,200,20,10,1000,500,5000,150,15,3,80,2,192.168.1.2,10.0.0.2,DoS\n"
        "22,6,50,5,2,250,100,5000,150,5,1,20,1,192.168.1.3,10.0.0.3,BENIGN\n"
    )
    p = tmp_path / "cicids2017_test.csv"
    p.write_text(data)
    return str(p)


@pytest.fixture
def sample_nslkdd_csv(tmp_path) -> str:
    """Crée un petit CSV NSL-KDD factice (sans header)."""
    row_normal = ",".join(["0"] * (len(NSL_KDD_COLUMNS) - 2)) + ",normal,20\n"
    row_attack = ",".join(["1"] * (len(NSL_KDD_COLUMNS) - 2)) + ",neptune,20\n"
    p = tmp_path / "nslkdd_test.csv"
    p.write_text(row_normal * 3 + row_attack * 2)
    return str(p)


# ─────────────────────────────────────────────────────────────────────────────
#  TESTS — DataLoader
# ─────────────────────────────────────────────────────────────────────────────

class TestDataLoaderInit:

    def test_valid_dataset_type(self):
        loader = DataLoader(dataset_type="CICIDS2017")
        assert loader.dataset_type == "CICIDS2017"

    def test_invalid_dataset_type(self):
        with pytest.raises(ValueError, match="Dataset non supporté"):
            DataLoader(dataset_type="UNKNOWN_DATASET")

    def test_label_column_set(self):
        loader = DataLoader(dataset_type="CICIDS2017")
        assert loader.label_column == "Label"

    def test_normal_label_set(self):
        loader = DataLoader(dataset_type="CICIDS2017")
        assert loader.normal_label == "BENIGN"

    def test_nslkdd_label(self):
        loader = DataLoader(dataset_type="NSL-KDD")
        assert loader.label_column == "class"


class TestLoadCSV:

    def test_load_cicids_csv(self, sample_cicids_csv):
        loader = DataLoader(dataset_type="CICIDS2017")
        df = loader.load_csv(sample_cicids_csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "Label" in df.columns

    def test_load_nslkdd_csv(self, sample_nslkdd_csv):
        loader = DataLoader(dataset_type="NSL-KDD")
        df = loader.load_csv(sample_nslkdd_csv)
        assert isinstance(df, pd.DataFrame)
        assert "class" in df.columns
        # difficulty_level doit être supprimé
        assert "difficulty_level" not in df.columns

    def test_nrows_limits_rows(self, sample_cicids_csv):
        loader = DataLoader(dataset_type="CICIDS2017")
        df = loader.load_csv(sample_cicids_csv, nrows=2)
        assert len(df) == 2

    def test_file_not_found(self):
        loader = DataLoader(dataset_type="CICIDS2017")
        with pytest.raises(FileNotFoundError):
            loader.load_csv("/nonexistent/path/file.csv")

    def test_columns_stripped(self, sample_cicids_csv):
        """Les colonnes CICIDS ne doivent pas avoir d'espaces."""
        loader = DataLoader(dataset_type="CICIDS2017")
        df = loader.load_csv(sample_cicids_csv)
        for col in df.columns:
            assert col == col.strip(), f"Colonne non strippée : '{col}'"

    def test_label_distribution_logged(self, sample_cicids_csv, caplog):
        loader = DataLoader(dataset_type="CICIDS2017")
        loader.load_csv(sample_cicids_csv)
        # La fonction de log ne lève pas d'exception

    def test_load_multiple_csv(self, tmp_path):
        """Charge et concatène plusieurs CSV."""
        data = (
            "Src IP,Dst IP,Protocol,Label\n"
            "1.1.1.1,2.2.2.2,6,BENIGN\n"
        )
        for i in range(3):
            (tmp_path / f"file_{i}.csv").write_text(data)

        loader = DataLoader(dataset_type="CICIDS2017")
        df = loader.load_multiple_csv(str(tmp_path))
        assert len(df) == 3

    def test_load_multiple_csv_empty_folder(self, tmp_path):
        loader = DataLoader(dataset_type="CICIDS2017")
        with pytest.raises(FileNotFoundError):
            loader.load_multiple_csv(str(tmp_path))
