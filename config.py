"""
Central Configuration Module for Deep Joint Source-Channel Communication (Deep JSCC).

Loads configuration from config.json and provides centralized, typed variables
used across model, dataset, training, testing, plots, and web app.
"""

import json
import os
from typing import Any, Dict, List, Tuple
import torch


CONFIG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def load_config_json(file_path: str = CONFIG_FILE_PATH) -> Dict[str, Any]:
    """Loads configuration from JSON file with fallback defaults if missing."""
    default_config: Dict[str, Any] = {
        "model": {
            "in_channels": 3,
            "channel_c": 16,
            "power": 1.0,
            "image_size": [32, 32],
        },
        "dataset": {
            "data_dir": "./data",
            "train_split": 0.7,
            "val_split": 0.3,
            "batch_size": 64,
            "test_batch_size": 64,
            "num_workers": 0,
            "random_seed": 42,
        },
        "channel": {
            "default_snr_db": 10.0,
            "test_snrs": [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
        },
        "training": {
            "epochs": 10,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "scheduler": "cosine",
        },
        "paths": {
            "checkpoint_dir": "./checkpoints",
            "best_model_path": "./checkpoints/best_jscc_model.pth",
            "outputs_dir": "./outputs",
        },
        "web": {
            "host": "0.0.0.0",
            "port": 8000,
        },
    }

    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                # Merge loaded into default_config
                for section, values in loaded.items():
                    if section in default_config and isinstance(values, dict):
                        default_config[section].update(values)
                    else:
                        default_config[section] = values
        except Exception as e:
            print(f"[config.py Warning] Could not parse {file_path}: {e}. Using defaults.")

    return default_config


# Load configuration
CONFIG = load_config_json()

# Model Parameters
MODEL_CONFIG: Dict[str, Any] = CONFIG.get("model", {})
IN_CHANNELS: int = int(MODEL_CONFIG.get("in_channels", 3))
CHANNEL_C: int = int(MODEL_CONFIG.get("channel_c", 16))
POWER_CONSTRAINT: float = float(MODEL_CONFIG.get("power", 1.0))
IMG_SIZE: Tuple[int, int] = tuple(MODEL_CONFIG.get("image_size", [32, 32]))  # type: ignore[assignment]

# Dataset Parameters
DATA_CONFIG: Dict[str, Any] = CONFIG.get("dataset", {})
DATA_DIR: str = str(DATA_CONFIG.get("data_dir", "./data"))
TRAIN_SPLIT: float = float(DATA_CONFIG.get("train_split", 0.7))
VAL_SPLIT: float = float(DATA_CONFIG.get("val_split", 0.3))
BATCH_SIZE: int = int(DATA_CONFIG.get("batch_size", 64))
TEST_BATCH_SIZE: int = int(DATA_CONFIG.get("test_batch_size", 64))
NUM_WORKERS: int = int(DATA_CONFIG.get("num_workers", 0))
RANDOM_SEED: int = int(DATA_CONFIG.get("random_seed", 42))

# Channel Parameters
CHANNEL_CONFIG: Dict[str, Any] = CONFIG.get("channel", {})
DEFAULT_SNR_DB: float = float(CHANNEL_CONFIG.get("default_snr_db", 10.0))
TEST_SNRS: List[float] = [float(s) for s in CHANNEL_CONFIG.get("test_snrs", [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0])]

# Training Parameters
TRAIN_CONFIG: Dict[str, Any] = CONFIG.get("training", {})
EPOCHS: int = int(TRAIN_CONFIG.get("epochs", 10))
LEARNING_RATE: float = float(TRAIN_CONFIG.get("learning_rate", 0.001))
WEIGHT_DECAY: float = float(TRAIN_CONFIG.get("weight_decay", 0.0))
SCHEDULER_TYPE: str = str(TRAIN_CONFIG.get("scheduler", "cosine"))

# Paths Parameters
PATHS_CONFIG: Dict[str, Any] = CONFIG.get("paths", {})
CHECKPOINT_DIR: str = str(PATHS_CONFIG.get("checkpoint_dir", "./checkpoints"))
BEST_MODEL_PATH: str = str(PATHS_CONFIG.get("best_model_path", "./checkpoints/best_jscc_model.pth"))
OUTPUTS_DIR: str = str(PATHS_CONFIG.get("outputs_dir", "./outputs"))

# Web Parameters
WEB_CONFIG: Dict[str, Any] = CONFIG.get("web", {})
WEB_HOST: str = str(WEB_CONFIG.get("host", "0.0.0.0"))
WEB_PORT: int = int(WEB_CONFIG.get("port", 8000))


def get_device() -> torch.device:
    """Returns the best available torch device: MPS (Apple Silicon), CUDA, or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_config_json(file_path: str = CONFIG_FILE_PATH) -> None:
    """Saves current configuration to file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2)


if __name__ == "__main__":
    print("=" * 60)
    print("Deep JSCC Central Configuration (config.py)")
    print("=" * 60)
    print(f"Device:             {get_device()}")
    print(f"Model In Channels:  {IN_CHANNELS}")
    print(f"Channel C:          {CHANNEL_C} (Bandwidth k = {CHANNEL_C * 8 * 8})")
    print(f"Power Constraint:   {POWER_CONSTRAINT}")
    print(f"Dataset Dir:        {DATA_DIR}")
    print(f"Train/Val Split:    {int(TRAIN_SPLIT * 100)}% / {int(VAL_SPLIT * 100)}%")
    print(f"Batch Size:         {BATCH_SIZE}")
    print(f"Training Epochs:    {EPOCHS} (LR: {LEARNING_RATE})")
    print(f"Nominal SNR:        {DEFAULT_SNR_DB} dB")
    print(f"Best Checkpoint:    {BEST_MODEL_PATH}")
    print(f"Outputs Dir:        {OUTPUTS_DIR}")
    print(f"Web Server Port:    {WEB_PORT}")
    print("=" * 60)
