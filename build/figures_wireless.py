"""Generate equations and schematic figures for the two wireless-perception Word docs:
  (1) the camera-reduction concept document, and
  (2) the ESPectre repository guide.

Equations use matplotlib mathtext (no external LaTeX). No emojis are used.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Wedge, Circle, Polygon

OUT = os.path.join(os.path.dirname(__file__), "assets_wireless")
os.makedirs(OUT, exist_ok=True)

INK = "#33373d"
SLATE = "#4a6b8a"
SOFT = "#5b7fa6"
BLUE = "#9ecae1"
GREEN = "#a8d5a2"
ORANGE = "#f4b183"
PURPLE = "#c4b7e0"
RED = "#e6a9a9"
GREY = "#d9dce1"
LGREY = "#eef1f4"
WALL = "#8a8f96"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "text.color": INK,
    "axes.edgecolor": "#b9bdc4",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p)


def render_eq(tex, name, fontsize=20):
    fig = plt.figure(figsize=(0.1, 0.1))
    t = fig.text(0, 0, tex, fontsize=fontsize, color=INK)
    fig.canvas.draw()
    bb = t.get_window_extent()
    w, h = bb.width / fig.dpi, bb.height / fig.dpi
    fig.set_size_inches(w + 0.2, h + 0.2)
    t.set_position((0.1 / (w + 0.2), 0.1 / (h + 0.2)))
    save(fig, name)


equations = {
    "eq_channel.png": r"$\mathbf{y}=\mathbf{H}\,\mathbf{x}+\mathbf{n}$",
    "eq_csi.png": r"$H(f_{k},t)=\sum_{p=1}^{P} a_{p}(t)\,e^{-j2\pi f_{k}\tau_{p}(t)}$",
    "eq_dyn.png": r"$H(t)=H_{\mathrm{static}}+\sum_{m} \alpha_{m}(t)\,e^{-j2\pi d_{m}(t)/\lambda}$",
    "eq_turbulence.png": r"$\tau_{t}=\sigma\left(|H_{1}|,\dots,|H_{K}|\right),\qquad V_{t}=\mathrm{Var}\left(\tau_{t-W+1},\dots,\tau_{t}\right)$",
    "eq_doppler.png": r"$f_{D}=\dfrac{2v}{\lambda}\cos\beta,\qquad f_{s}\;\geq\;2\,f_{D,\max}\quad(\mathrm{Nyquist})$",
    "eq_range.png": r"$\Delta R=\dfrac{c}{2B},\qquad \Delta\theta\approx\dfrac{\lambda}{D}$",
    "eq_dof.png": r"$N_{\mathrm{dof}}\;\approx\;\dfrac{A_{T}\,A_{R}}{(\lambda\,d)^{2}}$",
    "eq_fusion.png": r"$p(\mathbf{s}\mid \mathbf{z}_{\mathrm{cam}},\mathbf{z}_{\mathrm{rf}})\;\propto\;p(\mathbf{z}_{\mathrm{cam}}\mid \mathbf{s})\,p(\mathbf{z}_{\mathrm{rf}}\mid \mathbf{s})\,p(\mathbf{s})$",
    "eq_crossmodal.png": r"$\theta^{\star}=\arg\min_{\theta}\sum_{t\in\mathcal{O}} \mathcal{L}\!\left(f_{\theta}(\mathrm{CSI}_{t}),\;y_{t}^{\mathrm{cam}}\right)$",
    "eq_cameras.png": r"$C_{\min}^{\mathrm{vision}}=\sum_{r=1}^{R}\left\lceil A_{r}/A_{\mathrm{FoV}}\right\rceil\;\;\Longrightarrow\;\;1+R$",
}
for fname, tex in equations.items():
    render_eq(tex, fname)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def box(ax, x, y, w, h, text, fc, ec=None, fs=10, tc=INK, bold=False, rounded=0.02):
    ec = ec or fc
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.01,rounding_size={rounded}",
                       fc=fc, ec=ec, lw=1.4)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold" if bold else "normal", wrap=True)
    return p


def arrow(ax, x0, y0, x1, y1, color=SOFT, lw=2.0, style="-|>", mut=14, ls="-"):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                        mutation_scale=mut, color=color, lw=lw, linestyle=ls)
    ax.add_patch(a)
    return a


def camera(ax, x, y, color=SLATE, s=0.26):
    ax.add_patch(Rectangle((x - s, y - s * 0.7), 2 * s, 1.4 * s, fc=color, ec="white", lw=1))
    ax.add_patch(Circle((x, y), s * 0.45, fc="white", ec=color, lw=1))
    ax.add_patch(Circle((x, y), s * 0.22, fc=color, ec=color))


def node(ax, x, y, color=GREEN, label=None):
    ax.add_patch(Circle((x, y), 0.16, fc=color, ec=INK, lw=1.2))
    if label:
        ax.text(x, y - 0.42, label, ha="center", fontsize=8, color=INK)


# ---------------------------------------------------------------------------
# Figure 1: apartment - cameras only vs one camera + Wi-Fi
# ---------------------------------------------------------------------------
def fig_apartment():
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    def draw_plan(ax):
        # outer walls
        ax.add_patch(Rectangle((0, 0), 10, 7, fill=False, ec=WALL, lw=3))
        # internal walls dividing 3 rooms
        ax.plot([5, 5], [0, 7], color=WALL, lw=3)          # vertical split
        ax.plot([5, 10], [3.5, 3.5], color=WALL, lw=3)      # horizontal split (right side)
        # doors (gaps) drawn as white segments
        ax.plot([5, 5], [1.0, 2.0], color="white", lw=4)
        ax.plot([5, 5], [4.7, 5.7], color="white", lw=4)
        ax.plot([5, 6.0], [3.5, 3.5], color="white", lw=4)
        ax.text(2.5, 6.5, "Living room", ha="center", fontsize=9, color=SLATE)
        ax.text(7.5, 6.5, "Bedroom", ha="center", fontsize=9, color=SLATE)
        ax.text(7.5, 3.0, "Bathroom", ha="center", fontsize=9, color=SLATE)
        ax.set_xlim(-1.4, 11.4); ax.set_ylim(-0.6, 7.9); ax.axis("off"); ax.set_aspect("equal")

    # Left: cameras only
    ax = axes[0]
    draw_plan(ax)
    ax.set_title("Vision only: one camera per room (3-4 cameras)", fontsize=11, color=SLATE, fontweight="bold")
    # camera in living room with FoV wedge
    camera(ax, 1.0, 6.0)
    ax.add_patch(Wedge((1.0, 6.0), 4.0, 285, 350, fc=BLUE, alpha=0.35, ec="none"))
    # bedroom camera
    camera(ax, 9.0, 6.4)
    ax.add_patch(Wedge((9.0, 6.4), 3.6, 190, 260, fc=BLUE, alpha=0.35, ec="none"))
    # bathroom camera
    camera(ax, 9.4, 3.0)
    ax.add_patch(Wedge((9.4, 3.0), 3.2, 150, 220, fc=BLUE, alpha=0.35, ec="none"))
    # blind spot marker
    ax.text(2.7, 2.0, "blind\nspot", ha="center", fontsize=8, color="#a84a4a", style="italic")
    ax.add_patch(Circle((2.7, 3.0), 0.9, fill=False, ec="#a84a4a", lw=1.2, ls="--"))
    ax.text(5.0, -0.4, "Walls block each camera; every room needs its own.\nPrivacy concern in bedroom and bathroom.",
            ha="center", fontsize=8.5, color=INK, style="italic")

    # Right: one camera + Wi-Fi links
    ax = axes[1]
    draw_plan(ax)
    ax.set_title("One camera + Wi-Fi sensing in every room", fontsize=11, color=SLATE, fontweight="bold")
    camera(ax, 1.0, 6.0)
    ax.add_patch(Wedge((1.0, 6.0), 4.0, 285, 350, fc=BLUE, alpha=0.30, ec="none"))
    # router in living room
    node(ax, 2.2, 3.2, color=ORANGE)
    ax.text(2.2, 2.65, "Wi-Fi router", ha="center", fontsize=8, color=INK)
    # ESP32 nodes: one per room
    node(ax, 4.4, 5.2, color=GREEN); ax.text(4.4, 4.75, "ESP32", ha="center", fontsize=7.5, color=INK)
    node(ax, 8.6, 5.4, color=GREEN); ax.text(8.6, 4.95, "ESP32", ha="center", fontsize=7.5, color=INK)
    node(ax, 8.8, 1.6, color=GREEN); ax.text(8.8, 1.15, "ESP32", ha="center", fontsize=7.5, color=INK)
    # Wi-Fi links (dashed) through walls
    for (nx, ny) in [(4.4, 5.2), (8.6, 5.4), (8.8, 1.6)]:
        arrow(ax, 2.2, 3.2, nx, ny, color=ORANGE, lw=1.8, style="-", ls=(0, (4, 3)))
    ax.text(5.0, -0.4, "Wi-Fi passes through walls; each cheap link senses one room.\nOne camera kept only where identity/detail matters.",
            ha="center", fontsize=8.5, color=INK, style="italic")
    fig.tight_layout()
    save(fig, "fig_apartment.png")


# ---------------------------------------------------------------------------
# Figure 2: what each modality provides (capability ladder)
# ---------------------------------------------------------------------------
def fig_modalities():
    fig, ax = plt.subplots(figsize=(11.0, 4.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")
    tasks = ["Presence /\nmotion", "People\ncount", "Coarse\nlocation", "Activity /\ngesture",
             "Fall\ndetection", "Breathing /\nheart rate", "Body\npose", "Identity /\nappearance"]
    # camera can do all except through-wall/privacy; wifi can do left ones, pose only with rich HW+ML
    cam = [1, 1, 1, 1, 1, 0.4, 1, 1]
    wifi = [1, 0.85, 0.8, 0.8, 0.75, 0.7, 0.45, 0.1]
    x = np.arange(len(tasks)); w = 0.38
    ax.bar(x - w / 2, [c * 6 for c in cam], width=w, color=BLUE, label="Camera (in view)")
    ax.bar(x + w / 2, [v * 6 for v in wifi], width=w, color=GREEN, label="Wi-Fi sensing")
    for xi, v in zip(x, wifi):
        ax.text(xi + w / 2, v * 6 + 0.15, {1: "full", 0.85: "good", 0.8: "good", 0.75: "good",
                0.7: "ok", 0.45: "partial", 0.1: "no"}.get(v, ""), ha="center", fontsize=7.5, color="#3d7a4a")
    ax.set_xticks(x); ax.set_xticklabels(tasks, fontsize=8.5)
    ax.set_yticks([])
    ax.set_ylim(0, 7.2)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.set_title("What each sensor can deliver (taller = more capable)", fontsize=11, color=SLATE, fontweight="bold")
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_modalities.png")


# ---------------------------------------------------------------------------
# Figure 3: CSI tensor (antenna x subcarrier x time)
# ---------------------------------------------------------------------------
def fig_csi_tensor():
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    # draw a grid representing subcarriers x time, a few stacked for antennas
    rng = np.random.default_rng(1)
    for layer, dx, dy, c in [(2, 0.6, 0.6, GREY), (1, 0.3, 0.3, BLUE), (0, 0.0, 0.0, SOFT)]:
        base = rng.random((8, 12))
        for i in range(8):
            for j in range(12):
                val = base[i, j]
                ax.add_patch(Rectangle((1 + j * 0.55 + dx, 1 + i * 0.45 + dy), 0.5, 0.4,
                                       fc=(c if layer == 0 else c), alpha=0.35 + 0.5 * val if layer == 0 else 0.25,
                                       ec="white", lw=0.4))
    ax.text(1 + 6 * 0.55, 0.55, "subcarriers (frequency)  \u2192", ha="center", fontsize=9, color=INK)
    ax.text(0.7, 1 + 4 * 0.45, "antennas", rotation=90, va="center", fontsize=9, color=INK)
    ax.annotate("time \u2192", xy=(9.2, 5.6), xytext=(7.7, 6.2),
                arrowprops=dict(arrowstyle="-|>", color=INK), fontsize=9, color=INK)
    ax.text(11.4, 3.4, "Each cell:\namplitude |H|\nand phase \u2220H",
            ha="left", va="center", fontsize=8.5, color=SLATE)
    ax.set_title("CSI is the wireless 'pixel grid': antenna \u00d7 subcarrier \u00d7 time",
                 fontsize=10.5, color=SLATE, fontweight="bold")
    save(fig, "fig_csi_tensor.png")


# ---------------------------------------------------------------------------
# Figure 4: Fresnel-zone sensing geometry of a Tx-Rx link
# ---------------------------------------------------------------------------
def fig_geometry():
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    tx, rx = (1.4, 3.0), (10.6, 3.0)
    node(ax, *tx, color=ORANGE); ax.text(tx[0], tx[1] - 0.6, "Router (Tx)", ha="center", fontsize=9)
    node(ax, *rx, color=GREEN); ax.text(rx[0], rx[1] - 0.6, "ESP32 (Rx)", ha="center", fontsize=9)
    # concentric Fresnel ellipses
    from matplotlib.patches import Ellipse
    cx = (tx[0] + rx[0]) / 2
    for k, a in enumerate([4.7, 4.2, 3.6], start=1):
        b = 1.9 - 0.4 * (k - 1)
        ax.add_patch(Ellipse((cx, 3.0), 2 * a, 2 * b, fill=(k == 3), fc=BLUE, alpha=0.15, ec=SOFT, lw=1.2))
    ax.plot([tx[0], rx[0]], [tx[1], rx[1]], color=SOFT, lw=1.5, ls="--")
    # a walking person inside
    px = 6.0
    ax.add_patch(Circle((px, 3.7), 0.28, fc=RED, ec=INK, lw=1))
    ax.add_patch(Rectangle((px - 0.2, 2.5), 0.4, 1.0, fc=RED, ec=INK, lw=1))
    arrow(ax, px, 4.4, px + 1.1, 4.4, color="#a84a4a", lw=1.6)
    ax.text(px + 0.2, 4.7, "motion", fontsize=8.5, color="#a84a4a")
    ax.text(cx, 0.7, "A person crossing the link's Fresnel zones perturbs amplitude and phase.\nOne router-to-ESP32 link 'illuminates' one room.",
            ha="center", fontsize=8.5, color=INK, style="italic")
    ax.set_title("Sensing geometry of a single Wi-Fi link", fontsize=10.5, color=SLATE, fontweight="bold")
    save(fig, "fig_geometry.png")


# ---------------------------------------------------------------------------
# Figure 5: fusion architecture (1 camera + K Wi-Fi nodes -> hub)
# ---------------------------------------------------------------------------
def fig_fusion_arch():
    fig, ax = plt.subplots(figsize=(11.0, 3.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    box(ax, 0.3, 5.0, 2.4, 1.4, "Camera\n(1, key room)", BLUE, fs=9.5)
    box(ax, 0.3, 3.0, 2.4, 1.3, "ESP32 CSI\nRoom 1", GREEN, fs=9)
    box(ax, 0.3, 1.4, 2.4, 1.3, "ESP32 CSI\nRoom 2", GREEN, fs=9)
    box(ax, 0.3, -0.2, 2.4, 1.3, "ESP32 CSI\nRoom 3", GREEN, fs=9)
    box(ax, 3.6, 2.2, 2.4, 2.2, "Per-stream\nfeatures +\ndetection", LGREY, ec=SOFT, fs=9)
    box(ax, 6.7, 2.4, 2.4, 1.9, "Late fusion\n(Bayesian /\nlearned)", PURPLE, fs=9.5, bold=True)
    box(ax, 9.7, 2.5, 2.1, 1.7, "Events:\npresence, count,\nactivity, fall", ORANGE, fs=9, bold=True)
    for y in [5.7, 3.65, 2.05, 0.45]:
        arrow(ax, 2.7, y, 3.6, 3.3, color=SOFT, lw=1.5)
    arrow(ax, 6.0, 3.3, 6.7, 3.35, color=SOFT, lw=1.8)
    arrow(ax, 9.1, 3.35, 9.7, 3.35, color=SOFT, lw=1.8)
    ax.text(6.0, 0.6, "The camera supervises the Wi-Fi models where their views overlap (cross-modal training);\nafterwards Wi-Fi covers the rooms with no camera.",
            ha="center", fontsize=8.5, color=INK, style="italic")
    ax.set_title("Reference architecture: one camera, many cheap Wi-Fi sensors, one hub",
                 fontsize=10.5, color=SLATE, fontweight="bold")
    save(fig, "fig_fusion_arch.png")


# ---------------------------------------------------------------------------
# Figure 6: ESPectre processing pipeline
# ---------------------------------------------------------------------------
def fig_espectre_pipeline():
    fig, ax = plt.subplots(figsize=(11.5, 3.0))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 3.2); ax.axis("off")
    steps = [
        ("Raw CSI\n(64 subc.)", BLUE),
        ("Gain lock\n(AGC/FFT)", GREY),
        ("NBVI select\n12 subc.", GREEN),
        ("Turbulence\n\u03c3(|H|)", LGREY),
        ("Hampel +\nlow-pass", GREY),
        ("Moving\nvariance", LGREY),
        ("Adaptive\nthreshold", ORANGE),
        ("IDLE /\nMOTION", PURPLE),
    ]
    w = 1.42; gap = 0.22; x = 0.2
    for i, (txt, c) in enumerate(steps):
        box(ax, x, 1.0, w, 1.3, txt, c, fs=8.6, bold=(i == len(steps) - 1))
        if i < len(steps) - 1:
            arrow(ax, x + w, 1.65, x + w + gap, 1.65, color=SOFT, lw=1.7, mut=12)
        x += w + gap
    ax.text(6.7, 0.35, "MVS detector (default). An optional ML detector replaces the last stages with an MLP (9\u219232\u219216\u21921).",
            ha="center", fontsize=8.5, color=INK, style="italic")
    save(fig, "fig_espectre_pipeline.png")


# ---------------------------------------------------------------------------
# Figure 7: ESPectre multi-room deployment
# ---------------------------------------------------------------------------
def fig_espectre_arch():
    fig, ax = plt.subplots(figsize=(10.0, 3.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    box(ax, 0.4, 4.2, 2.4, 1.2, "ESP32\nRoom 1", GREEN, fs=9)
    box(ax, 0.4, 2.4, 2.4, 1.2, "ESP32\nRoom 2", GREEN, fs=9)
    box(ax, 0.4, 0.6, 2.4, 1.2, "ESP32\nRoom 3", GREEN, fs=9)
    box(ax, 4.6, 2.4, 3.0, 1.6, "Home Assistant\n(ESPHome API,\nauto-discovery)", BLUE, fs=9.5, bold=True)
    box(ax, 8.6, 2.5, 2.9, 1.4, "Dashboards,\nautomations,\nalerts", ORANGE, fs=9)
    for y in [4.8, 3.0, 1.2]:
        arrow(ax, 2.8, y, 4.6, 3.2, color=SOFT, lw=1.6)
    arrow(ax, 7.6, 3.2, 8.6, 3.2, color=SOFT, lw=1.8)
    ax.text(6.1, 0.35, "Each ESP32 exposes a motion binary sensor + movement score; all discovered automatically.",
            ha="center", fontsize=8.5, color=INK, style="italic")
    ax.set_title("ESPectre deployment: one ESP32 per room into Home Assistant",
                 fontsize=10.5, color=SLATE, fontweight="bold")
    save(fig, "fig_espectre_arch.png")


if __name__ == "__main__":
    fig_apartment()
    fig_modalities()
    fig_csi_tensor()
    fig_geometry()
    fig_fusion_arch()
    fig_espectre_pipeline()
    fig_espectre_arch()
    print("all wireless figures written to", OUT)
