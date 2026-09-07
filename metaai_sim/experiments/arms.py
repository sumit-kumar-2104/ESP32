"""Experimental arms: model + optimizer + readout choice.

Each arm is a self-contained ``train_and_eval`` callable that consumes
already-materialised (X_train, y_train, X_val, y_val, X_test, y_test) plus
the persisted ``sample_id`` list for the test set, and writes the following
under ``exp_dir``:

    metrics/epochs.csv          — per-epoch train_loss, train_acc, val_acc
    metrics/validation.json     — best + final validation metrics
    metrics/test.json           — locked-test metrics (from selected checkpoint)
    predictions/test_predictions.csv
    checkpoints/best.pt         — state_dict of the selected checkpoint
    checkpoints/last.pt         — state_dict at final epoch

The scaler / PCA / any learned transform lives OUTSIDE the arm — the runner
fits them on training data only and hands the arm the transformed arrays.
Arms therefore cannot leak.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import metrics as M


# ─── Arm registry ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ArmSpec:
    name: str
    feature_mode: str            # "raw" | "amp" | "dfs_spec" | ...
    dfs_bins: str = "full"       # only relevant for dfs_spec
    kind: str = "logreg"         # "logreg" | "digital_mlp" | "ota"
    hook: str = "none"           # "none" | "R1" | "R2"  (only when kind == "ota")
    epochs: int = 40
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.3
    complex_dim: int | None = None
    patience: int = 8
    qgrad_scale: float = 8.0
    lr_complex: float | None = None
    ste: str = "identity"
    extra: dict[str, Any] = field(default_factory=dict)


_REGISTRY: dict[str, ArmSpec] = {}


def register(spec: ArmSpec) -> ArmSpec:
    if spec.name in _REGISTRY:
        raise ValueError(f"arm {spec.name!r} already registered")
    _REGISTRY[spec.name] = spec
    return spec


def get_arm(name: str) -> ArmSpec:
    if name not in _REGISTRY:
        raise KeyError(f"unknown arm {name!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_arms() -> list[str]:
    return sorted(_REGISTRY)


# Canonical arms. Feature mode and hook are the only user-visible axes.
register(ArmSpec("logreg_raw", feature_mode="raw", kind="logreg", epochs=1))
register(ArmSpec("digital_mlp_raw", feature_mode="raw", kind="digital_mlp"))
register(ArmSpec("ota_raw_baseline", feature_mode="raw", kind="ota", hook="none"))
register(ArmSpec("ota_raw_R1", feature_mode="raw", kind="ota", hook="R1",
                 weight_decay=1e-3, dropout=0.4, complex_dim=1024, patience=6))
register(ArmSpec("ota_raw_R2", feature_mode="raw", kind="ota", hook="R2",
                 qgrad_scale=8.0, lr_complex=5e-3, ste="hardtanh"))

register(ArmSpec("logreg_dfs", feature_mode="dfs_spec", dfs_bins="full",
                 kind="logreg", epochs=1))
register(ArmSpec("digital_mlp_dfs", feature_mode="dfs_spec", dfs_bins="full",
                 kind="digital_mlp"))
register(ArmSpec("ota_dfs_baseline", feature_mode="dfs_spec", dfs_bins="full",
                 kind="ota", hook="none"))
register(ArmSpec("ota_dfs_R1", feature_mode="dfs_spec", dfs_bins="full",
                 kind="ota", hook="R1", weight_decay=1e-3, dropout=0.4,
                 complex_dim=512, patience=6))
register(ArmSpec("ota_dfs_R2", feature_mode="dfs_spec", dfs_bins="full",
                 kind="ota", hook="R2", qgrad_scale=8.0, lr_complex=5e-3,
                 ste="hardtanh"))


# ─── Training helpers ───────────────────────────────────────────────────────

def _write_epoch_row(path: Path, header: list[str], row: dict) -> None:
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)


def _select_device(preferred: str = "auto"):
    import torch
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError("cuda requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Arm implementations ────────────────────────────────────────────────────

def _train_logreg(
    spec: ArmSpec,
    X_train, y_train, X_val, y_val, X_test, y_test,
    *,
    sample_ids_test: Sequence[str],
    seed: int,
    n_classes: int,
    exp_dir: Path,
    class_index_to_gesture: dict[int, int],
) -> dict:
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(
        max_iter=1000, solver="lbfgs", random_state=seed,
    )
    clf.fit(X_train, y_train)
    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)
    try:
        test_probs = clf.predict_proba(X_test)
    except AttributeError:
        test_probs = None

    val_metrics = M.summarise_predictions(
        y_val, val_pred, class_index_to_gesture=class_index_to_gesture,
    )
    test_metrics = M.summarise_predictions(
        y_test, test_pred, class_index_to_gesture=class_index_to_gesture,
    )
    _write_epoch_row(
        exp_dir / "metrics" / "epochs.csv",
        header=["epoch", "train_loss", "train_acc", "val_acc"],
        row={"epoch": 1, "train_loss": float("nan"),
             "train_acc": float(clf.score(X_train, y_train)),
             "val_acc": val_metrics["accuracy"]},
    )
    M.write_json(exp_dir / "metrics" / "validation.json",
                 {"best": val_metrics, "final": val_metrics})
    M.write_json(exp_dir / "metrics" / "test.json", test_metrics)
    M.write_predictions_csv(
        exp_dir / "predictions" / "test_predictions.csv",
        sample_ids_test, y_test, test_pred, probs=test_probs,
    )
    return {
        "best_val_acc": val_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
    }


def _train_torch(
    spec: ArmSpec,
    build_model: Callable,
    build_optimizer: Callable,
    *,
    X_train, y_train, X_val, y_val, X_test, y_test,
    sample_ids_test: Sequence[str],
    seed: int,
    n_classes: int,
    device,
    exp_dir: Path,
    class_index_to_gesture: dict[int, int],
    complex_input: bool,
) -> dict:
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    model = build_model().to(device)
    opt = build_optimizer(model)
    ce = nn.CrossEntropyLoss()

    def _t(arr, dtype):
        return torch.tensor(np.asarray(arr), dtype=dtype, device=device)

    xtr, ytr = _t(X_train, torch.float32), _t(y_train, torch.long)
    xva, yva = _t(X_val, torch.float32), _t(y_val, torch.long)
    xte, yte = _t(X_test, torch.float32), _t(y_test, torch.long)
    n = xtr.shape[0]

    def _fwd(x):
        if complex_input:
            return model(torch.complex(x, torch.zeros_like(x)))
        return model(x)

    epochs_csv = exp_dir / "metrics" / "epochs.csv"
    header = ["epoch", "train_loss", "train_acc", "val_acc"]
    best_val, best_ep, stale = -1.0, 0, 0
    best_state = None

    for ep in range(1, spec.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        tot_loss = n_seen = n_correct = 0
        for i in range(0, n, spec.batch_size):
            idx = perm[i:i + spec.batch_size]
            xb, yb = xtr[idx], ytr[idx]
            opt.zero_grad(set_to_none=True)
            logits = _fwd(xb)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()
            tot_loss += float(loss.item()) * xb.size(0)
            n_seen += xb.size(0)
            n_correct += int((logits.argmax(1) == yb).sum().item())
        train_loss = tot_loss / max(n_seen, 1)
        train_acc = n_correct / max(n_seen, 1)

        model.eval()
        with torch.no_grad():
            logits = _fwd(xva)
            val_acc = float((logits.argmax(1) == yva).float().mean().item())

        _write_epoch_row(epochs_csv, header, {
            "epoch": ep, "train_loss": train_loss,
            "train_acc": train_acc, "val_acc": val_acc,
        })
        if val_acc > best_val + 1e-6:
            best_val, best_ep, stale = val_acc, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if spec.patience and stale >= spec.patience:
            break

    # Persist checkpoints.
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
               exp_dir / "checkpoints" / "last.pt")
    if best_state is not None:
        torch.save(best_state, exp_dir / "checkpoints" / "best.pt")
        model.load_state_dict(best_state)

    # Validation & test using the selected checkpoint.
    model.eval()
    with torch.no_grad():
        val_logits = _fwd(xva)
        val_pred = val_logits.argmax(1).cpu().numpy()
        test_logits = _fwd(xte)
        test_pred = test_logits.argmax(1).cpu().numpy()
        try:
            test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()
        except Exception:  # noqa: BLE001 — fallback if logits aren't calibrated
            test_probs = None

    val_metrics = M.summarise_predictions(
        yva.cpu().numpy(), val_pred,
        class_index_to_gesture=class_index_to_gesture,
    )
    final_metrics = M.summarise_predictions(
        yva.cpu().numpy(), val_logits.argmax(1).cpu().numpy(),
        class_index_to_gesture=class_index_to_gesture,
    )
    test_metrics = M.summarise_predictions(
        yte.cpu().numpy(), test_pred,
        class_index_to_gesture=class_index_to_gesture,
    )
    M.write_json(exp_dir / "metrics" / "validation.json",
                 {"best": val_metrics, "final": final_metrics,
                  "best_epoch": best_ep})
    M.write_json(exp_dir / "metrics" / "test.json", test_metrics)
    M.write_predictions_csv(
        exp_dir / "predictions" / "test_predictions.csv",
        sample_ids_test, yte.cpu().numpy(), test_pred, probs=test_probs,
    )
    return {
        "best_val_acc": val_metrics["accuracy"],
        "best_epoch": best_ep,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
    }


def _train_digital_mlp(spec, **kw):
    import torch.nn as nn
    import torch.optim as optim

    in_dim = kw["X_train"].shape[1]
    n_classes = kw["n_classes"]

    def build_model():
        return nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(spec.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def build_optimizer(model):
        return optim.Adam(model.parameters(), lr=spec.lr,
                          weight_decay=spec.weight_decay)

    return _train_torch(
        spec, build_model, build_optimizer,
        complex_input=False, **kw,
    )


def _train_ota(spec, **kw):
    """OTA ComplexLinear arm with optional R1/R2 hooks from fix_pipeline."""
    import torch.optim as optim
    import fix_pipeline as fp
    from models.linear_complex import ComplexLinear

    in_dim = kw["X_train"].shape[1]
    n_classes = kw["n_classes"]

    fp.disable_all_hooks()
    if spec.hook == "R1":
        fp.enable_hook_R1(
            wd=spec.weight_decay, dropout=spec.dropout,
            complex_dim=spec.complex_dim, patience=spec.patience,
        )
        def build_model():
            return fp.make_r1_ota_model(in_dim, n_classes)
        def build_optimizer(model):
            return optim.Adam(model.parameters(), lr=spec.lr,
                              weight_decay=spec.weight_decay)
    elif spec.hook == "R2":
        fp.enable_hook_R2(
            qgrad_scale=spec.qgrad_scale, lr_complex=spec.lr_complex,
            ste_kind=spec.ste,
        )
        def build_model():
            return fp.make_r2_ota_model(in_dim, n_classes)
        def build_optimizer(model):
            return optim.Adam(
                fp.r2_param_groups(model, base_lr=spec.lr),
                lr=spec.lr, weight_decay=spec.weight_decay,
            )
    else:
        def build_model():
            return ComplexLinear(in_dim, n_classes)
        def build_optimizer(model):
            return optim.Adam(model.parameters(), lr=spec.lr,
                              weight_decay=spec.weight_decay)

    return _train_torch(
        spec, build_model, build_optimizer,
        complex_input=True, **kw,
    )


def run_arm(
    spec: ArmSpec,
    *,
    X_train, y_train, X_val, y_val, X_test, y_test,
    sample_ids_test: Sequence[str],
    seed: int,
    n_classes: int,
    device_pref: str,
    exp_dir: Path,
    class_index_to_gesture: dict[int, int],
) -> dict:
    """Dispatch to the right arm implementation and return summary metrics."""
    exp_dir.mkdir(parents=True, exist_ok=True)
    common = dict(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        sample_ids_test=sample_ids_test,
        seed=seed, n_classes=n_classes,
        exp_dir=exp_dir,
        class_index_to_gesture=class_index_to_gesture,
    )
    if spec.kind == "logreg":
        return _train_logreg(spec, **common)
    device = _select_device(device_pref)
    common["device"] = device
    if spec.kind == "digital_mlp":
        return _train_digital_mlp(spec, **common)
    if spec.kind == "ota":
        return _train_ota(spec, **common)
    raise ValueError(f"unknown arm kind {spec.kind!r}")
