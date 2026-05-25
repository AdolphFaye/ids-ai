"""
utils/config_loader.py
───────────────────────────────────────────────────────────────
Chargement de la configuration YAML du pipeline.
Auteur : Alioune Badara Adolphe Faye
"""

import yaml
from pathlib import Path
from typing import Any


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """
    Charge le fichier de configuration YAML.

    Args:
        config_path: Chemin vers le fichier config.yaml

    Returns:
        Dictionnaire de configuration
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable : {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_nested(config: dict, *keys: str, default: Any = None) -> Any:
    """Accès sécurisé à une clé imbriquée dans le config."""
    val = config
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k, default)
        else:
            return default
    return val
