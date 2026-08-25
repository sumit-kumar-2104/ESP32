"""
Train the digital complex-valued model on MNIST or Widar3.0 BVP.
Stage 1: ideal (continuous) complex weights.
Stage 3A: --discrete flag trains the DiscreteNN baseline.
Stage 3D: --noise-train flag trains with noise injection.

Usage:
    python train.py                          # Stage 1 on MNIST (default)
    python train.py --dataset widar          # Stage 1 on Widar3.0
    python train.py --discrete               # DiscreteNN baseline (Stage 3A)
    python train.py --noise-train            # Noise-aware training (Stage 3D)
    python train.py --dataset widar --noise-train  # Noise-aware on Widar

Saves:
    results/stage1_weights.pt      — MNIST trained complex weight matrix
    results/widar_stage1_weights.pt — Widar trained complex weight matrix
    results/discrete_weights.pt    — DiscreteNN baseline weights
    results/noise_aware_weights.pt — noise-aware trained weights
    results/training_log.txt       — per-epoch train/test accuracy
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    BATCH_SIZE,
    EPOCHS,
    INPUT_DIM,
    LEARNING_RATE,
    MOMENTUM,
    NUM_CLASSES,
    RESULTS_DIR,
    TRAIN_SNR_DB,
    WIDAR_EPOCHS,
    WIDAR_LR,
    WIDAR_NUM_CLASSES,
    WIDAR_WEIGHT_DECAY,
    set_seed,
)
from data.loader import get_mnist_loaders
from data.widar_loader import get_widar_loaders
from models.linear_complex import ComplexLinear
from models.discrete_nn import DiscreteComplexLinear
from sim.sender import encode
from sim.channel import add_noise


def train(mode: str = "standard", dataset: str = "mnist"):
    """
    Train a model.
    mode: "standard" (Stage 1), "discrete" (Stage 3A), "noise" (Stage 3D)
    dataset: "mnist" or "widar"
    """
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print(f"[mode] {mode}")
    print(f"[dataset] {dataset}")

    # Data — resolve dimensions dynamically
    if dataset == "widar":
        train_loader, test_loader, feature_dim = get_widar_loaders(BATCH_SIZE)
        input_dim = feature_dim
        num_classes = WIDAR_NUM_CLASSES
        prefix = "widar_"
        paper_target = "~89.67%"
        epochs = WIDAR_EPOCHS * 2 if mode == "discrete" else WIDAR_EPOCHS
    else:
        train_loader, test_loader = get_mnist_loaders(BATCH_SIZE)
        input_dim = INPUT_DIM
        num_classes = NUM_CLASSES
        prefix = ""
        paper_target = "~92.75%"
        epochs = EPOCHS

    print(f"[model] INPUT_DIM={input_dim}, NUM_CLASSES={num_classes}, EPOCHS={epochs}")

    # Model
    if mode == "discrete":
        model = DiscreteComplexLinear(input_dim, num_classes).to(device)
        weights_path = RESULTS_DIR / f"{prefix}discrete_weights.pt"
        log_path = RESULTS_DIR / f"{prefix}discrete_training_log.txt"
    elif mode == "noise":
        model = ComplexLinear(input_dim, num_classes).to(device)
        weights_path = RESULTS_DIR / f"{prefix}noise_aware_weights.pt"
        log_path = RESULTS_DIR / f"{prefix}noise_training_log.txt"
    else:
        model = ComplexLinear(input_dim, num_classes).to(device)
        weights_path = RESULTS_DIR / f"{prefix}stage1_weights.pt"
        log_path = RESULTS_DIR / f"{prefix}training_log.txt"

    # Optimizer: Adam for Widar (stable on L2-normed 8k-dim), SGD for MNIST
    if dataset == "widar":
        lr = WIDAR_LR / 10 if mode == "discrete" else WIDAR_LR  # STE needs lower lr
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=WIDAR_WEIGHT_DECAY
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM
        )

    # Loss: we use cross-entropy on the magnitude outputs
    criterion = nn.CrossEntropyLoss()

    best_test_acc = 0.0

    with open(log_path, "w") as log_file:
        log_file.write("epoch,train_acc,test_acc,train_loss\n")

        for epoch in range(1, epochs + 1):
            # ─── Train ───
            model.train()
            correct = 0
            total = 0
            running_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                # Encode images to complex BPSK symbols
                x_complex = encode(images)

                # Forward pass
                y_mag = model(x_complex)

                # Stage 3D: inject noise during training (paper Eqns. 13-14)
                if mode == "noise":
                    # Add noise to the output (simulates hardware + env noise)
                    y_complex = torch.matmul(x_complex, model.complex_weight)
                    y_complex = add_noise(y_complex, snr_db=TRAIN_SNR_DB)
                    y_mag = torch.abs(y_complex)

                # Loss on magnitudes (treated as logits)
                loss = criterion(y_mag, labels)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accuracy
                preds = torch.argmax(y_mag, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                running_loss += loss.item() * labels.size(0)

            train_acc = 100.0 * correct / total
            train_loss = running_loss / total

            # ─── Test ───
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    x_complex = encode(images)
                    y_mag = model(x_complex)
                    preds = torch.argmax(y_mag, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            test_acc = 100.0 * correct / total

            # Log
            line = f"{epoch},{train_acc:.2f},{test_acc:.2f},{train_loss:.4f}"
            log_file.write(line + "\n")
            log_file.flush()

            if epoch % max(5, epochs // 12) == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:3d}/{epochs} | "
                    f"Train Acc: {train_acc:.2f}% | "
                    f"Test Acc: {test_acc:.2f}% | "
                    f"Loss: {train_loss:.4f}"
                )

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), weights_path)

    print(f"\n[{mode}] Training complete.")
    print(f"  Best test accuracy: {best_test_acc:.2f}%")
    if mode == "discrete":
        disc_target = "~82.33%" if dataset == "widar" else "~72.05%"
        print(f"  Paper target (DiscreteNN): {disc_target}")
    elif mode == "noise":
        print(f"  Paper target: noise-aware model improves low-SNR robustness")
    else:
        print(f"  Paper target:       {paper_target}")
    print(f"  Weights saved to:   {weights_path}")
    print(f"  Training log:       {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MetaAI Training")
    parser.add_argument("--discrete", action="store_true",
                        help="Train DiscreteNN baseline (Stage 3A)")
    parser.add_argument("--noise-train", action="store_true",
                        help="Train with noise injection (Stage 3D)")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "widar"],
                        help="Dataset: 'mnist' (default) or 'widar'")
    args = parser.parse_args()

    if args.discrete:
        train(mode="discrete", dataset=args.dataset)
    elif args.noise_train:
        train(mode="noise", dataset=args.dataset)
    else:
        train(mode="standard", dataset=args.dataset)
