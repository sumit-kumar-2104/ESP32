"""
Stage 5.1 — In-domain benchmark: train all three baselines on the same
Widar i.i.d. split and compare accuracy/params/time.

Usage:
    python benchmark/run_benchmark.py
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import set_seed, RESULTS_DIR, BATCH_SIZE, WIDAR_LR, WIDAR_WEIGHT_DECAY
from data.widar_loader import get_widar_loaders
from benchmark.models import MetaAILinear, DigitalLinear, MLP2Layer

# ─── Device ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] {device}")
if device.type == "cuda":
    print(f"[device] GPU: {torch.cuda.get_device_name(0)}")

# ─── Hyperparameters ──────────────────────────────────────────────────────────
EPOCHS = 200
LR = 1e-3
SEED = 42


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_and_eval(model, train_loader, test_loader, epochs=EPOCHS, lr=LR):
    """Train a model and return (train_acc, test_acc, elapsed_seconds)."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WIDAR_WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train_loader:
            # images shape: (batch, 1, 8000) -> squeeze to (batch, 8000)
            x = images.squeeze(1).to(device)
            labels = labels.to(device)
            logits = model(x)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    train_acc = evaluate(model, train_loader)
    test_acc = evaluate(model, test_loader)
    return train_acc, test_acc


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            x = images.squeeze(1).to(device)
            labels = labels.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    set_seed(SEED)
    print(f"\n{'='*60}")
    print("STAGE 5.1 — IN-DOMAIN BENCHMARK (i.i.d. split, single date)")
    print(f"{'='*60}\n")

    # Load data (uses config: single date 20181109-VS, iid split)
    train_loader, test_loader, feature_dim = get_widar_loaders(BATCH_SIZE)
    num_classes = 6
    print(f"[benchmark] feature_dim={feature_dim}, num_classes={num_classes}")
    print(f"[benchmark] epochs={EPOCHS}, lr={LR}, weight_decay={WIDAR_WEIGHT_DECAY}")
    print(f"[benchmark] device={device}\n")

    # Define models
    models_dict = {
        "MetaAI-Linear": MetaAILinear(feature_dim, num_classes),
        "DigitalLinear": DigitalLinear(feature_dim, num_classes),
        "MLP-2layer": MLP2Layer(feature_dim, num_classes),
    }

    results = []
    for name, model in models_dict.items():
        print(f"  Training {name} ...")
        set_seed(SEED)  # reset seed for fair comparison
        n_params = count_params(model)
        t0 = time.time()
        train_acc, test_acc = train_and_eval(model, train_loader, test_loader)
        elapsed = time.time() - t0
        results.append({
            "model": name,
            "device": str(device),
            "train_acc": f"{train_acc*100:.2f}%",
            "test_acc": f"{test_acc*100:.2f}%",
            "params": n_params,
            "seconds": f"{elapsed:.1f}",
        })
        print(f"    {name}: train={train_acc*100:.2f}%, test={test_acc*100:.2f}%, "
              f"params={n_params}, time={elapsed:.1f}s")

    # ─── Results table ────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS (in-domain, i.i.d. split)")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print()

    # Save CSV
    csv_path = RESULTS_DIR / "benchmark_indomain.csv"
    df.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")

    # ─── Bar chart ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    model_names = [r["model"] for r in results]
    test_accs = [float(r["test_acc"].strip("%")) for r in results]
    train_accs = [float(r["train_acc"].strip("%")) for r in results]

    x = np.arange(len(model_names))
    width = 0.35
    bars1 = ax.bar(x - width/2, train_accs, width, label="Train Acc", color="#4C72B0")
    bars2 = ax.bar(x + width/2, test_accs, width, label="Test Acc", color="#DD8452")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Stage 5.1: In-Domain Benchmark (Widar3.0, single date 20181109-VS)")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_ylim(0, 105)
    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    plt.tight_layout()
    png_path = RESULTS_DIR / "benchmark_indomain.png"
    fig.savefig(png_path, dpi=150)
    print(f"[saved] {png_path}")
    plt.close()

    print("\n[DONE] Stage 5.1 complete.")


if __name__ == "__main__":
    main()
