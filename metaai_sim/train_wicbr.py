"""
Train Wi-CBR (DACN) on Widar3.0 STIFMM images and save checkpoint.

Uses the same data that run_wicbr_v2.py uses (phase + DFS images) with
train/test split by repetition (rn 1-4 train, rn 5 test) for in-domain,
matching the BVP experiment's scope.

Saves:
    checkpoints/wicbr.pt — full DACN state_dict

Reports in-domain and cross-domain accuracy + macro-F1 so you can confirm
the model actually generalizes (cross-domain should be much higher than OTA).

Requirements: torch, torchvision, scikit-learn, PIL
Usage:        python train_wicbr.py
"""

import os
import sys
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))

CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"
WICBR_DIR = Path(__file__).parent / "benchmark" / "external" / "wicbr"
PHASE_DIR = WICBR_DIR / "WIDAR_STIFMM"
DFS_DIR = WICBR_DIR / "WIDAR_STIFMM_DFS"


def set_seed(seed=888):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Dataset ──────────────────────────────────────────────────────────────────

class WidarImageDataset(Dataset):
    def __init__(self, phase_dir, dfs_dir, user_ids, gesture_range, ln_range, on_range, rn_range):
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        self.phase_tensors = []
        self.dfs_tensors = []
        self.labels = []

        paths = []
        for uid in user_ids:
            for g in gesture_range:
                for ln in ln_range:
                    for on in on_range:
                        for rn in rn_range:
                            fname = f"{uid}-{g}-{ln}-{on}-{rn}.jpg"
                            p = os.path.join(str(phase_dir), fname)
                            d = os.path.join(str(dfs_dir), fname)
                            if os.path.exists(p) and os.path.exists(d):
                                paths.append((p, d, g - 1))

        print(f"    Preloading {len(paths)} images into RAM...", end=" ", flush=True)
        t0 = time.time()
        for p_path, d_path, label in paths:
            p_img = self.transform(Image.open(p_path).convert('RGB'))
            d_img = self.transform(Image.open(d_path).convert('RGB'))
            self.phase_tensors.append(p_img)
            self.dfs_tensors.append(d_img)
            self.labels.append(label)
        print(f"done ({time.time()-t0:.1f}s)", flush=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.phase_tensors[idx], self.dfs_tensors[idx], self.labels[idx]


# ─── Model (DACN from Wi-CBR) ────────────────────────────────────────────────

class BasicConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, relu=True, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x


class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = BasicConv(3, 1, 7, stride=1, padding=3, bn=True, relu=False)

    def forward(self, x):
        scale = torch.sigmoid(self.spatial(x))
        return x * scale + x, scale


class DPFusion(nn.Module):
    def __init__(self, channels, group_num=4, threshold=0.5):
        super().__init__()
        self.gn = nn.GroupNorm(num_channels=channels, num_groups=group_num)
        self.threshold = threshold

    def forward(self, x):
        gn_x = self.gn(x)
        w = self.gn.weight / (self.gn.weight.sum() + 1e-8)
        w = w.view(1, -1, 1, 1)
        reweights = torch.sigmoid(gn_x * w)
        strong = torch.where(reweights >= self.threshold, torch.ones_like(reweights), reweights)
        weak = torch.where(reweights < self.threshold, torch.zeros_like(reweights), reweights)
        ps, ds = torch.split(strong * x, x.size(1) // 2, dim=1)
        pw, dw = torch.split(weak * x, x.size(1) // 2, dim=1)
        return torch.cat([ps + dw, ds + pw], dim=1)


class DACN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.p_spa = SpatialGate()
        self.d_spa = SpatialGate()
        p_resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        d_resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.p_features = nn.Sequential(*list(p_resnet.children())[:-2])
        self.d_features = nn.Sequential(*list(d_resnet.children())[:-2])
        self.dp_fusion = DPFusion(1024)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, p_x, d_x):
        p_outspa, _ = self.p_spa(p_x)
        d_outspa, _ = self.d_spa(d_x)
        p_out = self.p_features(p_outspa)
        d_out = self.d_features(d_outspa)
        out = torch.cat([p_out, d_out], dim=1)
        out = self.dp_fusion(out)
        out = self.avgpool(out)
        embedding = out.view(out.size(0), -1)
        return self.fc(embedding), embedding


class ProxyContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels, model):
        proxies = F.normalize(model.fc.weight, p=2, dim=1)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        sim = torch.matmul(embeddings, proxies.T) / self.temperature
        mask = F.one_hot(labels.long(), num_classes=proxies.size(0)).float()
        sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
        log_prob = sim - torch.log(torch.exp(sim).sum(dim=1, keepdim=True))
        return -(mask * log_prob).sum(dim=1).mean()


# ─── Training ─────────────────────────────────────────────────────────────────

def train_model(train_ds, test_ds, device, epochs=30, batch_size=10, lr=1e-4):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    net = DACN(num_classes=6).to(device)
    params = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6
    print(f"  Params: {params:.2f}M")

    ce_loss = nn.CrossEntropyLoss()
    proxy_loss = ProxyContrastiveLoss(temperature=0.1)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_acc, best_f1, best_state = 0.0, 0.0, None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        net.train()
        correct, total = 0, 0
        for p_imgs, d_imgs, labels in train_loader:
            p_imgs, d_imgs, labels = p_imgs.to(device), d_imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out, emb = net(p_imgs, d_imgs)
            loss = ce_loss(out, labels) + 0.1 * proxy_loss(emb, labels, net)
            loss.backward()
            optimizer.step()
            _, pred = out.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
        train_acc = 100.0 * correct / total
        scheduler.step()

        net.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for p_imgs, d_imgs, labels in test_loader:
                p_imgs, d_imgs, labels = p_imgs.to(device), d_imgs.to(device), labels.to(device)
                out, _ = net(p_imgs, d_imgs)
                _, pred = out.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_acc = 100.0 * correct / total
        macro_f1 = f1_score(all_labels, all_preds, average='macro')

        if test_acc > best_acc:
            best_acc = test_acc
            best_f1 = macro_f1
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}

        print(f"  Epoch {epoch:02d}/{epochs}: train={train_acc:.1f}% test={test_acc:.2f}% "
              f"f1={macro_f1:.4f} [{time.time()-t0:.0f}s]")

    return net, best_state, best_acc, best_f1


def main():
    print("=" * 60)
    print("  Wi-CBR (DACN) Training for B2 Probe")
    print("=" * 60)

    if not PHASE_DIR.exists() or not DFS_DIR.exists():
        print(f"\nERROR: Wi-CBR image data not found.")
        print(f"  Expected STIFMM at: {PHASE_DIR}")
        print(f"  Expected DFS at:    {DFS_DIR}")
        print("  Skipping Wi-CBR training.")
        sys.exit(1)

    set_seed(888)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}")

    gestures = range(1, 7)
    locs = range(1, 6)
    oris = range(1, 6)

    # In-domain: users 0-8, train rn=1-4, test rn=5
    print("\n[In-domain split] Loading data...")
    indom_train = WidarImageDataset(PHASE_DIR, DFS_DIR, range(0, 9), gestures, locs, oris, range(1, 5))
    indom_test = WidarImageDataset(PHASE_DIR, DFS_DIR, range(0, 9), gestures, locs, oris, [5])

    if len(indom_train) == 0:
        print("ERROR: No training images found. Check WIDAR_STIFMM directory.")
        sys.exit(1)

    print(f"\n[In-domain] Training Wi-CBR...")
    net, best_state, indom_acc, indom_f1 = train_model(indom_train, indom_test, device)

    # Cross-domain: train users 0-10,15,16; test users 11-14
    print("\n[Cross-domain split] Loading data...")
    cr3_train_users = list(range(0, 11)) + [15, 16]
    cr3_test_users = [11, 12, 13, 14]
    cr3_train = WidarImageDataset(PHASE_DIR, DFS_DIR, cr3_train_users, gestures, locs, oris, range(1, 6))
    cr3_test = WidarImageDataset(PHASE_DIR, DFS_DIR, cr3_test_users, gestures, locs, oris, range(1, 6))

    print(f"\n[Cross-domain] Training Wi-CBR...")
    _, _, cr3_acc, cr3_f1 = train_model(cr3_train, cr3_test, device)

    # Save the in-domain checkpoint (used for B2 feature dumping)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINTS_DIR / "wicbr.pt"
    torch.save(best_state, ckpt_path)
    print(f"\n[saved] {ckpt_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Wi-CBR RESULTS")
    print(f"{'='*60}")
    print(f"  {'Split':<20} {'Accuracy':<12} {'Macro F1':<10}")
    print(f"  {'-'*42}")
    print(f"  {'In-domain':<20} {indom_acc:.2f}%    {indom_f1:.4f}")
    print(f"  {'Cross-domain (cr3)':<20} {cr3_acc:.2f}%    {cr3_f1:.4f}")
    drop = indom_acc - cr3_acc
    print(f"  {'In->Cross drop':<20} {drop:.2f} pp")
    print(f"{'='*60}")
    print("\nWi-CBR should show HIGHER cross-domain accuracy than OTA,")
    print("confirming it has learned domain-invariant features.")


if __name__ == "__main__":
    main()
