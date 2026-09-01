"""
Configuration for MetaAI simulation.
All hyperparameters and paths in one place.
Reference: Feng et al., "Enabling Over-the-Air AI for Edge Computing via
Metasurface-Driven Physical Neural Networks", ACM SIGCOMM 2025.
"""

import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ─── Seed (reproducibility) ───────────────────────────────────────────────────
SEED = 42

# ─── Dataset selection ─────────────────────────────────────────────────────────
DATASET = "mnist"      # "mnist" or "widar"

# ─── Training hyperparameters (from paper) ────────────────────────────────────
LEARNING_RATE = 8e-3
MOMENTUM = 0.95
BATCH_SIZE = 64
EPOCHS = 60
WIDAR_EPOCHS = 200     # Enough for convergence on 8000-dim L2-normed features

# ─── Model dimensions ─────────────────────────────────────────────────────────
# For MNIST: INPUT_DIM=784, NUM_CLASSES=10
# For Widar: INPUT_DIM set dynamically from loaded BVP features, NUM_CLASSES=6
INPUT_DIM = 784        # default for MNIST; overridden at runtime for Widar
NUM_CLASSES = 10       # default for MNIST; overridden at runtime for Widar

# ─── Widar3.0 parameters (paper Table 1) ──────────────────────────────────────
WIDAR_NUM_CLASSES = 6
WIDAR_DATE_SCOPE = "single"  # "single" (20181109-VS, in-domain) or "pooled"
WIDAR_SPLIT = "iid"          # "iid" (random 90/10, faithful to reference) or "rep" (reps 1-16/17-20)
WIDAR_LR = 1e-3              # Adam lr for Widar (lower than MNIST SGD lr)
WIDAR_WEIGHT_DECAY = 1e-3    # L2 regularization to prevent overfitting (96k params / 6748 samples)

# ─── Stage 2: Metasurface quantization ────────────────────────────────────────
N_META_ATOMS = 256            # number of metasurface cells (M in paper Eqn. 4)
USE_QUANTIZATION = False      # toggle quantization in evaluate.py
PHASE_STATES = [0, np.pi / 2, np.pi, 3 * np.pi / 2]  # 2-bit phase states

# ─── Stage 3: Robustness mechanisms (paper Section 3.5) ───────────────────────
# (B) CDFA Clock Synchronization
USE_CDFA = False
SYNC_ERROR_US = 4.0         # sync delay error to simulate, microseconds
GAMMA_SHAPE = 2.0           # Gamma dist shape for timing-error injection (σ)
GAMMA_SCALE = 1.0           # Gamma dist scale (β)

# (C) Multipath Cancellation (paper Section 3.2, Fig. 8)
USE_MULTIPATH = False
ENV_CHANNEL_TAPS = 3        # number of static environmental multipath taps
SAMPLES_PER_SYMBOL = 8      # intra-symbol sampling points for cancellation

# (D) Noise-Aware Training (paper Eqns. 13–14)
USE_NOISE_TRAINING = False
TRAIN_SNR_DB = 10.0         # deliberately low SNR during training
EVAL_SNR_DB_LIST = [5, 10, 15, 20, 25, 30]  # sweep for noise robustness plot

# ─── Data directory resolver ──────────────────────────────────────────────────
def get_data_dir() -> Path:
    """
    Resolve data directory (never inside repo):
      1) METAAI_DATA_DIR env var if set
      2) ~/.cache/metaai_data  (per-machine default)
    """
    env = os.environ.get("METAAI_DATA_DIR")
    if env:
        data_dir = Path(env)
    else:
        data_dir = Path.home() / ".cache" / "metaai_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def print_data_info(data_dir: Path) -> None:
    """Print resolved data path and download status."""
    mnist_exists = (data_dir / "MNIST").exists()
    widar_exists = (data_dir / "widar3" / "BVP").exists()
    print(f"[data] using METAAI_DATA_DIR = {data_dir}")
    print(f"[data]   MNIST downloaded: {'yes' if mnist_exists else 'no'}")
    print(f"[data]   Widar3.0 BVP:     {'yes' if widar_exists else 'no'}")


# ─── Results directory ────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Logs directory ───────────────────────────────────────────────────────────
LOGS_DIR = Path(__file__).parent / "logs"


class _Tee:
    """Duplicate a stream to several targets (console + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def setup_logging(script_name: str) -> Path:
    """Tee stdout/stderr to a timestamped file under logs/ and return its path."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"{script_name}_{ts}.log"
    fh = open(log_path, "a", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    print(f"[log] writing timestamped log to {log_path}")
    return log_path


def get_device() -> "torch.device":
    """Return the compute device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_device() -> "torch.device":
    """Print and return the compute device."""
    dev = get_device()
    if dev.type == "cuda":
        print(f"[device] using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[device] using CPU")
    return dev

# ─── Seed setter ──────────────────────────────────────────────────────────────
def set_seed(seed: int = SEED) -> None:
    """Set global random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[seed] global seed = {seed}")
