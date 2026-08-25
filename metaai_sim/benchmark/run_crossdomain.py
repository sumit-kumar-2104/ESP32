"""
Stage 5.2 — Cross-domain benchmark: leave-one-date-out protocol.
Trains on all valid dates EXCEPT one, tests on the held-out date.
Compares in-domain (i.i.d.) vs cross-domain (leave-one-out) accuracy.

Usage:
    python benchmark/run_crossdomain.py
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import set_seed, RESULTS_DIR, BATCH_SIZE, WIDAR_WEIGHT_DECAY
from data.widar_loader import (
    VALID_DATES, IN_DOMAIN_DATE, NUM_GESTURES, BVP_KEY, T_FIXED,
    BUGGY_STEMS, _get_widar_dir, _is_buggy, _normalize_per_frame, _interpolate_time,
)
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


def load_date(date_folder: str):
    """Load all valid BVP samples for a single date. Returns (features, labels) numpy arrays."""
    import scipy.io
    widar_dir = _get_widar_dir()
    date_path = widar_dir / "BVP" / date_folder
    if not date_path.exists():
        return None, None

    features_list = []
    labels_list = []

    for mat_file in sorted(date_path.rglob("*.mat")):
        stem = mat_file.stem
        if _is_buggy(stem):
            continue
        parts = stem.split("-")
        if len(parts) < 5:
            continue
        try:
            gesture_id = int(parts[1])
        except ValueError:
            continue
        if gesture_id < 1 or gesture_id > NUM_GESTURES:
            continue
        try:
            mat_data = scipy.io.loadmat(str(mat_file))
            if BVP_KEY not in mat_data:
                continue
            bvp = mat_data[BVP_KEY]
            if bvp.ndim != 3 or bvp.shape[0] != 20 or bvp.shape[1] != 20:
                continue
            if bvp.shape[2] == 0:
                continue
        except Exception:
            continue

        bvp_norm = _normalize_per_frame(bvp)
        bvp_interp = _interpolate_time(bvp_norm, T_FIXED)
        features_list.append(bvp_interp.flatten().astype(np.float32))
        labels_list.append(gesture_id - 1)

    if not features_list:
        return None, None
    return np.stack(features_list), np.array(labels_list, dtype=np.int64)


def make_loaders(X_train, y_train, X_test, y_test, batch_size=BATCH_SIZE):
    """Create DataLoaders from numpy arrays (matching widar_loader format)."""
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_and_eval(model, train_loader, test_loader):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WIDAR_WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x_batch, labels in train_loader:
            x_batch, labels = x_batch.to(device), labels.to(device)
            logits = model(x_batch)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    train_acc = evaluate(model, train_loader)
    test_acc = evaluate(model, test_loader)
    return train_acc, test_acc


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x_batch, labels in loader:
            x_batch, labels = x_batch.to(device), labels.to(device)
            preds = model(x_batch).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    set_seed(SEED)
    print(f"\n{'='*70}")
    print("STAGE 5.2 — CROSS-DOMAIN BENCHMARK (leave-one-date-out)")
    print(f"{'='*70}\n")

    # Load all dates
    print("[data] Loading all valid dates...")
    date_data = {}
    for d in VALID_DATES:
        X, y = load_date(d)
        if X is not None:
            date_data[d] = (X, y)
            print(f"  {d}: {len(y)} samples")
        else:
            print(f"  {d}: MISSING/EMPTY — skipping")

    valid_dates = list(date_data.keys())
    print(f"\n[data] {len(valid_dates)} dates loaded, "
          f"{sum(len(v[1]) for v in date_data.values())} total samples")

    feature_dim = date_data[valid_dates[0]][0].shape[1]
    num_classes = 6

    # ─── In-domain baseline (pooled i.i.d.) ───────────────────────────────────
    print(f"\n--- In-domain (pooled i.i.d. 90/10 split) ---")
    X_all = np.concatenate([date_data[d][0] for d in valid_dates])
    y_all = np.concatenate([date_data[d][1] for d in valid_dates])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.1, random_state=SEED, stratify=y_all
    )
    train_loader_iid, test_loader_iid = make_loaders(X_tr, y_tr, X_te, y_te)

    model_factories = {
        "MetaAI-Linear": lambda: MetaAILinear(feature_dim, num_classes),
        "DigitalLinear": lambda: DigitalLinear(feature_dim, num_classes),
        "MLP-2layer": lambda: MLP2Layer(feature_dim, num_classes),
    }

    indomain_results = {}
    for name, factory in model_factories.items():
        set_seed(SEED)
        model = factory()
        print(f"  Training {name} (in-domain)...")
        _, test_acc = train_and_eval(model, train_loader_iid, test_loader_iid)
        indomain_results[name] = test_acc
        print(f"    {name} in-domain test acc: {test_acc*100:.2f}%")

    # ─── Cross-domain (leave-one-date-out) ────────────────────────────────────
    # Use a subset of dates to keep runtime reasonable (5 folds)
    fold_dates = valid_dates[:5]
    print(f"\n--- Cross-domain (leave-one-date-out, {len(fold_dates)} folds) ---")
    print(f"  Held-out dates: {fold_dates}")

    crossdomain_accs = {name: [] for name in model_factories}

    for held_out in fold_dates:
        print(f"\n  Fold: held-out = {held_out}")
        # Train on all except held-out
        train_dates = [d for d in valid_dates if d != held_out]
        X_train = np.concatenate([date_data[d][0] for d in train_dates])
        y_train = np.concatenate([date_data[d][1] for d in train_dates])
        X_test, y_test = date_data[held_out]

        train_loader, test_loader = make_loaders(X_train, y_train, X_test, y_test)

        for name, factory in model_factories.items():
            set_seed(SEED)
            model = factory()
            _, test_acc = train_and_eval(model, train_loader, test_loader)
            crossdomain_accs[name].append(test_acc)
            print(f"    {name}: {test_acc*100:.2f}%")

    # ─── Summary table ────────────────────────────────────────────────────────
    rows = []
    for name in model_factories:
        cd_mean = np.mean(crossdomain_accs[name])
        cd_std = np.std(crossdomain_accs[name])
        rows.append({
            "model": name,
            "in_domain_acc": f"{indomain_results[name]*100:.2f}%",
            "cross_domain_mean": f"{cd_mean*100:.2f}%",
            "cross_domain_std": f"{cd_std*100:.2f}%",
            "drop": f"{(indomain_results[name] - cd_mean)*100:.2f}pp",
        })

    df = pd.DataFrame(rows)
    print(f"\n{'='*70}")
    print("CROSS-DOMAIN RESULTS SUMMARY")
    print(f"{'='*70}")
    print(df.to_string(index=False))
    print()

    # Save CSV
    csv_path = RESULTS_DIR / "benchmark_crossdomain.csv"
    df.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")

    # ─── Grouped bar chart ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    model_names = list(model_factories.keys())
    id_accs = [indomain_results[n] * 100 for n in model_names]
    cd_accs = [np.mean(crossdomain_accs[n]) * 100 for n in model_names]

    x = np.arange(len(model_names))
    width = 0.35
    bars1 = ax.bar(x - width/2, id_accs, width, label="In-Domain (i.i.d.)", color="#4C72B0")
    bars2 = ax.bar(x + width/2, cd_accs, width, label="Cross-Domain (leave-one-out)", color="#C44E52")

    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Stage 5.2: In-Domain vs Cross-Domain (Widar3.0, all dates)")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_ylim(0, 105)
    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    plt.tight_layout()
    png_path = RESULTS_DIR / "benchmark_crossdomain.png"
    fig.savefig(png_path, dpi=150)
    print(f"[saved] {png_path}")
    plt.close()

    print("\n[DONE] Stage 5.2 complete.")


if __name__ == "__main__":
    main()
