"""Equations and schematic figures for the camera-to-wireless identity ideas doc.

Renders matplotlib-mathtext equations (no LaTeX) and a few schematic figures.
No emojis are used.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

OUT = os.path.join(os.path.dirname(__file__), "assets_ideas")
os.makedirs(OUT, exist_ok=True)

INK = "#33373d"; SLATE = "#4a6b8a"; SOFT = "#5b7fa6"; BLUE = "#9ecae1"
GREEN = "#a8d5a2"; ORANGE = "#f4b183"; PURPLE = "#c4b7e0"; RED = "#e6a9a9"
GREY = "#d9dce1"; LGREY = "#eef1f4"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
                     "figure.facecolor": "white", "axes.facecolor": "white"})


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("wrote", p)


def render_eq(tex, name, fontsize=20):
    fig = plt.figure(figsize=(0.1, 0.1))
    t = fig.text(0, 0, tex, fontsize=fontsize, color=INK)
    fig.canvas.draw(); bb = t.get_window_extent()
    w, h = bb.width / fig.dpi, bb.height / fig.dpi
    fig.set_size_inches(w + 0.2, h + 0.2)
    t.set_position((0.1 / (w + 0.2), 0.1 / (h + 0.2)))
    save(fig, name)


equations = {
    "eq_embed.png": r"$\mathbf{z}=f_{\theta}(\mathbf{x})\in\mathbb{R}^{d}$",
    "eq_match.png": r"$k^{\star}=\mathrm{arg\,min}_{k}\; d\!\left(\mathbf{z}_{\mathrm{rf}},\,\mathbf{z}^{\,k}_{\mathrm{cam}}\right)$",
    "eq_triplet.png": r"$\mathcal{L}_{\mathrm{tri}}=\sum\left[\;\|\mathbf{z}_{a}-\mathbf{z}_{p}\|^{2}-\|\mathbf{z}_{a}-\mathbf{z}_{n}\|^{2}+m\;\right]_{+}$",
    "eq_align.png": r"$\min_{\theta,\phi}\;\sum_{i}\left\|\,f_{\theta}(\mathrm{CSI}_{i})-g_{\phi}(\mathrm{video}_{i})\,\right\|^{2}$",
    "eq_assoc.png": r"$\hat{A}=\mathrm{arg\,min}_{A}\sum_{i,j} A_{ij}\,C_{ij},\qquad C_{ij}=-\log p\!\left(\mathrm{match}\mid \mathbf{z}^{\mathrm{cam}}_{i},\mathbf{z}^{\mathrm{rf}}_{j}\right)$",
    "eq_verify.png": r"$\mathrm{same\ person}\;\Leftrightarrow\; d\!\left(\mathbf{z}_{\mathrm{rf}},\mathbf{z}_{\mathrm{cam}}\right)<\tau$",
}
for fname, tex in equations.items():
    render_eq(tex, fname)


def box(ax, x, y, w, h, text, fc, ec=None, fs=10, tc=INK, bold=False, rounded=0.03):
    ec = ec or fc
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.01,rounding_size={rounded}",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1, color=SOFT, lw=2.0, style="-|>", mut=14, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=mut,
                                 color=color, lw=lw, linestyle=ls))


def person(ax, x, y, color=RED, label=None, s=1.0):
    ax.add_patch(Circle((x, y + 0.32 * s), 0.14 * s, fc=color, ec=INK, lw=1))
    ax.add_patch(Rectangle((x - 0.11 * s, y - 0.28 * s), 0.22 * s, 0.55 * s, fc=color, ec=INK, lw=1))
    if label:
        ax.text(x, y - 0.5 * s, label, ha="center", fontsize=8, color=INK, fontweight="bold")


def camera(ax, x, y, color=SLATE, s=0.22):
    ax.add_patch(Rectangle((x - s, y - s * 0.7), 2 * s, 1.4 * s, fc=color, ec="white", lw=1))
    ax.add_patch(Circle((x, y), s * 0.42, fc="white", ec=color, lw=1))
    ax.add_patch(Circle((x, y), s * 0.2, fc=color, ec=color))


def node(ax, x, y, color=GREEN):
    ax.add_patch(Circle((x, y), 0.15, fc=color, ec=INK, lw=1.2))


# ---------------------------------------------------------------------------
# Figure 1: the identity handoff concept (enroll -> signature -> wireless track)
# ---------------------------------------------------------------------------
def fig_handoff():
    fig, ax = plt.subplots(figsize=(12.0, 4.4))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 6.2); ax.axis("off")

    # Stage 1: enrollment gate with camera
    ax.add_patch(Rectangle((0.2, 1.2), 3.0, 4.2, fill=False, ec=GREY, lw=2))
    ax.text(1.7, 5.05, "1. Enrolment gate", ha="center", fontsize=10, color=SLATE, fontweight="bold")
    camera(ax, 1.0, 4.2)
    person(ax, 2.1, 3.0, color=BLUE, label="known\nperson")
    ax.text(1.7, 1.55, "face + gait + body shape", ha="center", fontsize=8, color=INK, style="italic")

    # Stage 2: signature
    box(ax, 3.7, 2.7, 2.2, 1.4, "2. Per-person\nsignature\n(embedding z)", PURPLE, fs=9.5, bold=True)
    arrow(ax, 3.2, 3.4, 3.7, 3.4, color=SOFT, lw=2.0)

    # Stage 3: three camera-free rooms with wireless nodes
    ax.text(9.9, 5.55, "3. Camera-free rooms monitored by wireless", ha="center", fontsize=10,
            color=SLATE, fontweight="bold")
    rooms_x = [6.7, 9.3, 11.9]
    labels = ["Room A", "Room B", "Room C"]
    for rx, lb in zip(rooms_x, labels):
        ax.add_patch(Rectangle((rx - 1.1, 1.2), 2.2, 3.6, fill=False, ec=GREY, lw=2))
        ax.text(rx, 4.5, lb, ha="center", fontsize=9, color=SLATE)
        node(ax, rx + 0.7, 3.9, color=GREEN)
        ax.text(rx + 0.7, 4.15, "RF", ha="center", fontsize=7, color=INK)
    # same person moving across rooms, re-identified by wireless
    for rx in rooms_x:
        person(ax, rx - 0.2, 2.6, color=BLUE, label="ID\nmatched")
    arrow(ax, 5.9, 3.4, 6.7 - 1.1, 3.4, color=SOFT, lw=2.0)
    for i in range(len(rooms_x) - 1):
        arrow(ax, rooms_x[i] + 1.1, 2.6, rooms_x[i + 1] - 1.1, 2.6, color=ORANGE, lw=1.6, ls=(0, (4, 3)))
    ax.text(9.9, 0.7, "The wireless signature measured live in each room is matched to the enrolled signature, "
                      "so the same identity is tracked without any camera in these rooms.",
            ha="center", fontsize=8.5, color=INK, style="italic")
    save(fig, "fig_handoff.png")


# ---------------------------------------------------------------------------
# Figure 2: shared cross-modal embedding space
# ---------------------------------------------------------------------------
def fig_embedding():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    ax.text(5, 7.5, "Shared embedding space", ha="center", fontsize=11, color=SLATE, fontweight="bold")
    rng = np.random.default_rng(4)
    centers = [(2.6, 5.2), (6.8, 5.6), (4.6, 2.4)]
    cols = [BLUE, GREEN, ORANGE]
    names = ["Person 1", "Person 2", "Person 3"]
    for (cx, cy), col, nm in zip(centers, cols, names):
        # camera gallery points (circles)
        for _ in range(5):
            p = (cx + rng.normal(0, 0.45), cy + rng.normal(0, 0.45))
            ax.add_patch(Circle(p, 0.13, fc=col, ec="white", lw=0.6))
        ax.text(cx, cy + 1.15, nm, ha="center", fontsize=8.5, color=INK)
    # an RF query near person 2 cluster
    q = (6.5, 5.2)
    ax.plot(*q, marker="*", ms=20, color="#b5651d", mec=INK, mew=0.8)
    ax.text(q[0] + 0.2, q[1] - 0.5, "live RF\nquery", fontsize=8.5, color="#b5651d")
    ax.annotate("nearest = Person 2", xy=q, xytext=(2.2, 6.9),
                arrowprops=dict(arrowstyle="-|>", color=INK), fontsize=8.5, color=INK)
    # legend
    ax.add_patch(Circle((1.0, 0.7), 0.13, fc=GREY, ec="white"))
    ax.text(1.25, 0.7, "camera gallery embeddings", va="center", fontsize=8, color=INK)
    ax.plot(6.2, 0.7, marker="*", ms=14, color="#b5651d", mec=INK)
    ax.text(6.5, 0.7, "wireless query embedding", va="center", fontsize=8, color=INK)
    save(fig, "fig_embedding.png")


# ---------------------------------------------------------------------------
# Figure 3: three-phase workflow (enroll / train / deploy)
# ---------------------------------------------------------------------------
def fig_phases():
    fig, ax = plt.subplots(figsize=(11.5, 2.7))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis("off")
    phases = [
        ("Enrol", "Camera captures face,\ngait and body shape;\nWiFi/RF records\nsimultaneously", BLUE),
        ("Train", "Learn embeddings so\npaired camera and RF\nsignatures land at the\nsame point (metric learning)", PURPLE),
        ("Deploy", "Camera removed from\nprivate rooms; wireless\nmatches live signatures\nto enrolled identities", GREEN),
    ]
    w = 3.5; gap = 0.6; x = 0.4
    for i, (t, b, c) in enumerate(phases):
        box(ax, x, 0.4, w, 2.2, "", c, rounded=0.06)
        ax.text(x + w / 2, 2.25, t, ha="center", fontsize=12, color=INK, fontweight="bold")
        ax.text(x + w / 2, 1.25, b, ha="center", fontsize=8.8, color=INK)
        if i < 2:
            arrow(ax, x + w, 1.5, x + w + gap, 1.5, color=SOFT, lw=2.2)
        x += w + gap
    save(fig, "fig_phases.png")


if __name__ == "__main__":
    fig_handoff()
    fig_embedding()
    fig_phases()
    print("all idea figures written to", OUT)
