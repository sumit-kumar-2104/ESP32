"""
Stage 5.3d — Unified master comparison table.
Two groups: our BVP models vs Wi-CBR SOTA (native inputs).
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.size'] = 11

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

# ─── Data ─────────────────────────────────────────────────────────────────────
rows = [
    # Group A: Our BVP models (5.1 in-domain i.i.d. + 5.2 leave-one-date-out)
    {"group": "A: Over-the-air (BVP)", "model": "MetaAI-Linear",
     "in_domain_acc": 80.67, "cross_domain_acc": 39.05, "params": "96K",
     "notes": "BVP (20,20,20), date split"},
    {"group": "A: Over-the-air (BVP)", "model": "DigitalLinear",
     "in_domain_acc": 81.33, "cross_domain_acc": 41.62, "params": "48K",
     "notes": "BVP (20,20,20), date split"},
    {"group": "A: Over-the-air (BVP)", "model": "MLP-2layer",
     "in_domain_acc": 83.47, "cross_domain_acc": 46.18, "params": "2.0M",
     "notes": "BVP (20,20,20), date split"},

    # Group B: Wi-CBR SOTA (native Phase+DFS, user/env split)
    {"group": "B: Cross-domain SOTA (native)", "model": "Wi-CBR (DACN)",
     "in_domain_acc": 99.56, "cross_domain_acc": 98.00, "params": "22.4M",
     "notes": "Phase+DFS 224x224, user/env split"},
]

df = pd.DataFrame(rows)
df["drop"] = df["in_domain_acc"] - df["cross_domain_acc"]
df["drop_str"] = df["drop"].apply(lambda x: f"{x:.1f}pp")

# ─── CSV ──────────────────────────────────────────────────────────────────────
csv_path = os.path.join(RESULTS_DIR, "benchmark_master.csv")
df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")

# ─── Print table ──────────────────────────────────────────────────────────────
print("\n" + "="*95)
print("  STAGE 5.3 — MASTER BENCHMARK TABLE")
print("="*95)
print(f"  {'Group':<30} {'Model':<16} {'In-dom':<9} {'Cross':<9} {'Drop':<8} {'Params':<8} {'Notes'}")
print(f"  {'-'*93}")
for _, r in df.iterrows():
    print(f"  {r['group']:<30} {r['model']:<16} {r['in_domain_acc']:.2f}%  {r['cross_domain_acc']:.2f}%  {r['drop_str']:<8} {r['params']:<8} {r['notes']}")

print(f"\n  CAVEAT: Same underlying Widar3.0 recordings, DIFFERENT input modality")
print(f"          (BVP vs Phase+DFS) and DIFFERENT split axis (date vs user/env).")
print(f"          This is a paradigm-level comparison, not a same-input head-to-head.")
print(f"\n  HEADLINE: Our over-the-air BVP models collapse cross-domain")
print(f"            (~81-83% -> ~39-46%, drop 37-42pp).")
print(f"            Wi-CBR's purpose-built cross-domain model barely drops")
print(f"            (99.56% -> 98.00%, drop 1.56pp).")
print("="*95)

# ─── Bar chart ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

models = df["model"].tolist()
x = np.arange(len(models))
width = 0.32

colors_in = ["#2196F3", "#2196F3", "#2196F3", "#FF9800"]
colors_cross = ["#1565C0", "#1565C0", "#1565C0", "#E65100"]

bars1 = ax.bar(x - width/2, df["in_domain_acc"], width, label="In-domain acc",
               color=colors_in, edgecolor="white", linewidth=0.5)
bars2 = ax.bar(x + width/2, df["cross_domain_acc"], width, label="Cross-domain acc",
               color=colors_cross, edgecolor="white", linewidth=0.5)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel("Accuracy (%)")
ax.set_title("Stage 5.3 — Master Benchmark: Over-the-air BVP vs Cross-domain SOTA\n"
             "(CAVEAT: different input modalities and split axes — paradigm-level comparison)",
             fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.set_ylim(0, 115)
ax.legend(loc="upper left")

# Add group separators
ax.axvline(x=2.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.text(1, 110, "GROUP A: Over-the-air (BVP, date split)", ha='center', fontsize=9,
        fontstyle='italic', color='#1565C0')
ax.text(3, 110, "GROUP B: SOTA\n(native, user/env split)", ha='center', fontsize=9,
        fontstyle='italic', color='#E65100')

# Drop annotations
for i, (_, r) in enumerate(df.iterrows()):
    mid_y = (r["in_domain_acc"] + r["cross_domain_acc"]) / 2
    ax.annotate(f"-{r['drop']:.1f}pp", xy=(i + width/2, r["cross_domain_acc"]),
                xytext=(i + width/2 + 0.2, mid_y),
                fontsize=8, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=0.8))

plt.tight_layout()
png_path = os.path.join(RESULTS_DIR, "benchmark_master.png")
plt.savefig(png_path, dpi=150)
print(f"\nSaved: {png_path}")
plt.close()
