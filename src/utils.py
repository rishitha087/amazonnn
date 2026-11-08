import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

def get_logger(name: str = "Amazon_MLOps") -> logging.Logger:
    """Returns a configured logger writing to both stdout and logs/pipeline.log."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console Handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File Handler
        try:
            from src.config import LOG_FILE_PATH, LOGS_DIR
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(LOG_FILE_PATH), encoding="utf-8")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception:
            pass
            
    return logger

def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """Safely saves a dictionary to a JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(str(filepath), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def print_banner(title: str, width: int = 70) -> None:
    """Prints a styled terminal banner."""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width + "\n")
