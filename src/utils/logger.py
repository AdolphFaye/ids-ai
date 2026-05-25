"""
utils/logger.py
───────────────────────────────────────────────────────────────
Logging centralisé pour tout le pipeline.
Auteur : Alioune Badara Adolphe Faye
"""

import sys
from loguru import logger


def get_logger(name: str = "IDS-Pipeline"):
    """Retourne un logger configuré avec le nom du module."""
    logger.remove()
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>" + name + "</cyan> | "
            "<level>{message}</level>"
        ),
        level="INFO",
        colorize=True,
    )
    logger.add(
        "logs/pipeline.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
    )
    return logger
