"""Generate rendered equations and schematic figures for the MetaAI explainer.

MetaAI paper: "Enabling Over-the-Air AI for Edge Computing via
Metasurface-Driven Physical Neural Networks" (SIGCOMM '25, Feng et al.).

Equations are typeset with matplotlib mathtext (no external LaTeX needed).
All figures use a light, presentable palette and contain no emojis.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

OUT = os.path.join(os.path.dirname(__file__), "assets_metaai")
os.makedirs(OUT, exist_ok=True)

# Light, presentable palette (shared with the document / deck)
INK = "#33373d"
SLATE = "#4a6b8a"
SOFT = "#5b7fa6"
BLUE = "#9ecae1"
GREEN = "#a8d5a2"
ORANGE = "#f4b183"
PURPLE = "#c4b7e0"
PINK = "#e6a9b8"
GREY = "#d9dce1"
LGREY = "#eef1f4"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "text.color": INK,
    "axes.edgecolor": "#b9bdc4",
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.titlecolor": INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p)


# ---------------------------------------------------------------------------
# Equation rendering via matplotlib mathtext
# ---------------------------------------------------------------------------
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
    # Linear neural network as a matrix-vector product
    "eq_lnn.png": r"$Y = W X,\qquad Y\in\mathbb{C}^{R},\ \ W\in\mathbb{C}^{R\times U},\ \ X\in\mathbb{C}^{U}$",
    # Sequential decomposition of one output
    "eq_decomp.png": r"$y_{1}=w_{1,1}\,x_{1}+w_{1,2}\,x_{2}+\cdots+w_{1,U}\,x_{U}=\sum_{i=1}^{U} w_{1,i}\,x_{i}$",
    # The wireless channel is a linear operator
    "eq_channel.png": r"$y(t) = H(t)\cdot x(t)$",
    # Core MetaAI computation (Eqn. 3)
    "eq_core.png": r"$y_{r}=\left|\;\sum_{i=0}^{U} H_{r}(t_{i})\cdot x_{i}\;\right|$",
    # Metasurface channel model (Eqn. 4)
    "eq_mts.png": r"$H_{mts}=\alpha_{p}\sum_{m=1}^{M} e^{\,j\phi_{m}^{p}}\;e^{\,j\phi_{m}}$",
    # Design goal (Eqn. 5)
    "eq_goal.png": r"$H_{des}=H_{mts}$",
    # Far-field path length (Eqn. 6)
    "eq_farfield.png": r"$d_{m,Rx}=d_{1,Rx}-(m-1)\,d_{s}\cos(\theta)$",
    # Configuration optimisation (Eqn. 7)
    "eq_config.png": r"$\Phi=\mathrm{arg\,min}_{\phi_{m}}\;\left|\,H_{mts}-H_{des}\,\right|,\qquad \Phi=[\phi_{1},\phi_{2},\dots,\phi_{M}]$",
    # Multipath-aware optimisation (Eqn. 8)
    "eq_multipath.png": r"$\Phi=\mathrm{arg\,min}_{\phi_{m}}\;\left|\,H_{mts}-(H_{des}-H_{e})\,\right|$",
    # Multi-sensor per-sensor output (Eqn. 11)
    "eq_sensor.png": r"$y_{r}^{\,s}=\sum_{i=1}^{U_{s}} H_{r}^{\,s}(t_{i}^{\,s})\cdot x_{i}^{\,s}$",
    # Multi-sensor fusion (Eqn. 12)
    "eq_fusion.png": r"$y_{r}^{\,multi}=\left|\;\sum_{s=1}^{N_{s}} y_{r}^{\,s}\;\right|$",
    # Noise-inclusive model (Eqn. 13)
    "eq_noise.png": r"$y_{r}=\left|\;\sum_{i=0}^{U}\left[\,H_{mts}(t_{i})+\mathcal{N}_{d}\,\right]\cdot x_{i}+\mathcal{N}_{e}\;\right|$",
    # Noise refactored as pre-disturbed signal (Eqn. 14)
    "eq_noise2.png": r"$y_{r}=\left|\;\sum_{i=0}^{U} H_{mts}(t_{i})\cdot\left(x_{i}+\hat{\mathcal{N}}_{d}\right)+\mathcal{N}_{e}\;\right|,\quad \hat{\mathcal{N}}_{d}=\dfrac{x_{i}}{H_{mts}(t_{i})}\,\mathcal{N}_{d}$",
    # Wave number
    "eq_wavenumber.png": r"$\phi_{m}^{p}=k_{0}\,(d_{Tx,m}+d_{m,Rx}),\qquad k_{0}=\dfrac{2\pi}{\lambda}$",
    # Subcarrier / antenna parallel loss (Eqn. 9)
    "eq_parallel.png": r"$loss=-\sum_{k=1}^{K} y_{k}\,\log\!\left(\left|\sum_{i=1}^{U} x_{i,k}\sum_{m=1}^{M} e^{\,j(\phi_{i,m}+\phi_{m,k}^{p})}\right|\right)$",
}
for fname, tex in equations.items():
    render_eq(tex, fname)


# ---------------------------------------------------------------------------
# Helper drawing primitives
# ---------------------------------------------------------------------------
def box(ax, x, y, w, h, text, fc, ec=None, fs=10, tc=INK, bold=False, rounded=0.03):
    ec = ec or fc
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.01,rounding_size={rounded}",
                       fc=fc, ec=ec, lw=1.4)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold" if bold else "normal", wrap=True)
    return p


def arrow(ax, x0, y0, x1, y1, color=SOFT, lw=2.0, style="-|>", mut=14):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                        mutation_scale=mut, color=color, lw=lw)
    ax.add_patch(a)
    return a


# ---------------------------------------------------------------------------
# Figure 1: three computing paradigms
# ---------------------------------------------------------------------------
def fig_paradigms():
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    titles = ["(a) Conventional digital", "(b) Physical neural network", "(c) MetaAI"]
    for ax in axes:
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    # (a) transmit then compute on server
    ax = axes[0]
    box(ax, 0.5, 6.5, 3.0, 2.0, "IoT device\n(sensor)", BLUE, fs=10)
    box(ax, 6.0, 6.5, 3.4, 2.0, "Edge server\nGPU / CPU", GREY, fs=10)
    arrow(ax, 3.5, 7.5, 6.0, 7.5)
    ax.text(4.75, 8.15, "raw data", ha="center", fontsize=8.5, color=INK)
    box(ax, 6.0, 2.6, 3.4, 1.7, "AI runs\non server", ORANGE, fs=10)
    arrow(ax, 7.7, 6.5, 7.7, 4.3)
    box(ax, 6.0, 0.4, 3.4, 1.3, "Result: 'Cat'", GREEN, fs=10, bold=True)
    arrow(ax, 7.7, 2.6, 7.7, 1.7)
    ax.text(2.0, 5.4, "communication and\ncomputation are\nseparate costs",
            ha="center", fontsize=8.5, color="#a04a4a", style="italic")

    # (b) PNN co-processor
    ax = axes[1]
    box(ax, 0.5, 6.5, 3.0, 2.0, "IoT device", BLUE, fs=10)
    box(ax, 4.3, 6.3, 2.2, 2.4, "PNN\n(mask /\nmetasurface)", PURPLE, fs=9)
    box(ax, 7.2, 6.5, 2.3, 2.0, "Edge\nserver", GREY, fs=9.5)
    arrow(ax, 3.5, 7.5, 4.3, 7.5)
    arrow(ax, 6.5, 7.5, 7.2, 7.5)
    ax.text(5.0, 4.6, "wave lights up the\nstructure = fast compute,\nbut data still fully\ntransmitted first",
            ha="center", fontsize=8.5, color="#a04a4a", style="italic")
    box(ax, 3.5, 0.6, 3.0, 1.3, "Result: 'Cat'", GREEN, fs=10, bold=True)

    # (c) MetaAI
    ax = axes[2]
    box(ax, 0.4, 6.5, 2.8, 2.0, "IoT device\ntransmits", BLUE, fs=9.5)
    box(ax, 6.6, 6.5, 3.0, 2.0, "Edge server\nreceives result", GREEN, fs=9.5, bold=True)
    # metasurface in the middle of the path
    box(ax, 3.7, 5.6, 2.3, 3.4, "Meta-\nsurface\n(computes\nin the air)", PURPLE, fs=9, bold=True)
    arrow(ax, 3.2, 7.5, 3.7, 7.5)
    arrow(ax, 6.0, 7.5, 6.6, 7.5)
    ax.text(5.0, 3.9, "communication AND\ncomputation happen\ntogether, in one pass",
            ha="center", fontsize=8.5, color="#3d7a4a", style="italic", fontweight="bold")
    box(ax, 3.4, 1.0, 3.2, 1.3, "Result: 'Cat'", GREEN, fs=10, bold=True)

    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=11, color=SLATE, fontweight="bold", pad=6)
    fig.tight_layout()
    save(fig, "fig_paradigms.png")


# ---------------------------------------------------------------------------
# Figure 2: sequential decomposition (parallel NN -> time slots)
# ---------------------------------------------------------------------------
def fig_sequential():
    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")

    # left: parallel multiply-accumulate
    ax.text(2.5, 7.5, "Parallel neural network", ha="center", fontsize=11,
            color=SLATE, fontweight="bold")
    xs = [0.6, 0.6, 0.6]
    ys = [5.6, 4.3, 3.0]
    labels = ["x1", "x2", "x3"]
    for y, lb in zip(ys, labels):
        box(ax, 0.4, y, 1.0, 0.9, lb, BLUE, fs=10)
    box(ax, 3.4, 4.0, 1.3, 1.2, "y1", GREEN, fs=11, bold=True)
    for y, w in zip(ys, ["w11", "w12", "w13"]):
        arrow(ax, 1.4, y + 0.45, 3.4, 4.6, color=SOFT, lw=1.6)
        ax.text(2.3, y + 0.45 + (4.6 - (y + 0.45)) * 0.35, w, fontsize=8, color=INK)
    ax.text(2.5, 1.9, "all inputs multiplied\nand summed at once",
            ha="center", fontsize=8.5, style="italic", color=INK)

    # arrow to the right
    arrow(ax, 5.1, 4.2, 6.2, 4.2, color=ORANGE, lw=2.4, mut=18)
    ax.text(5.65, 4.7, "linear\n=>", ha="center", fontsize=9, color="#b5651d")

    # right: time slots
    ax.text(9.0, 7.5, "Equivalent sequential computation", ha="center", fontsize=11,
            color=SLATE, fontweight="bold")
    slot_x = [6.6, 8.3, 10.0]
    prods = ["w11 . x1", "w12 . x2", "w13 . x3"]
    for i, (sx, pr) in enumerate(zip(slot_x, prods)):
        box(ax, sx, 4.6, 1.5, 1.1, pr, LGREY, ec=SOFT, fs=9)
        ax.text(sx + 0.75, 6.05, f"time slot {i+1}", ha="center", fontsize=8.5, color=SLATE)
        if i < 2:
            ax.text(sx + 1.6, 5.15, "+", ha="center", va="center", fontsize=16, color=ORANGE)
    ax.text(11.55, 5.15, "=", ha="center", va="center", fontsize=16, color=INK)
    box(ax, 10.0, 2.9, 1.5, 1.1, "y1", GREEN, fs=11, bold=True)
    arrow(ax, 10.75, 4.6, 10.75, 4.0, color=SOFT, lw=1.6)
    ax.text(9.0, 1.9, "one product per transmitted symbol,\naccumulated over time in software",
            ha="center", fontsize=8.5, style="italic", color=INK)

    save(fig, "fig_sequential.png")


# ---------------------------------------------------------------------------
# Figure 3: MetaAI over-the-air pipeline
# ---------------------------------------------------------------------------
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11.0, 3.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")

    box(ax, 0.3, 2.6, 2.1, 1.9, "Input data\n(image /\nsignal)", BLUE, fs=9)
    box(ax, 2.8, 2.6, 2.0, 1.9, "Encode +\nmodulate\n(e.g. QAM)", GREY, fs=9)
    # transmitter antenna
    ax.text(5.6, 4.9, "Tx", ha="center", fontsize=10, color=SLATE, fontweight="bold")
    box(ax, 5.2, 2.9, 0.9, 1.3, ">>", LGREY, ec=SOFT, fs=12)
    # metasurface
    box(ax, 6.7, 1.4, 1.9, 4.4, "Meta-\nsurface\nH(t)\napplies\nweights", PURPLE, fs=9.5, bold=True)
    # receiver
    ax.text(9.7, 4.9, "Rx", ha="center", fontsize=10, color=SLATE, fontweight="bold")
    box(ax, 9.3, 2.9, 0.9, 1.3, ">>", LGREY, ec=SOFT, fs=12)
    box(ax, 10.6, 2.6, 1.2, 1.9, "Result\n'Cat'", GREEN, fs=9.5, bold=True)

    arrow(ax, 2.4, 3.55, 2.8, 3.55)
    arrow(ax, 4.8, 3.55, 5.2, 3.55)
    # curved wave from Tx to MTS to Rx
    arrow(ax, 6.1, 3.55, 6.7, 3.55, color=ORANGE, lw=2.2)
    arrow(ax, 8.6, 3.55, 9.3, 3.55, color=ORANGE, lw=2.2)
    arrow(ax, 10.2, 3.55, 10.6, 3.55)

    ax.text(7.65, 0.7, "multiply over the air", ha="center", fontsize=8.5,
            color="#6a4a9a", style="italic")
    ax.text(5.6, 1.9, "sequential symbols", ha="center", fontsize=8, color=INK, style="italic")
    ax.text(9.7, 1.9, "accumulate = y_r", ha="center", fontsize=8, color=INK, style="italic")
    save(fig, "fig_pipeline.png")


# ---------------------------------------------------------------------------
# Figure 4: metasurface with meta-atoms and 2-bit phase states
# ---------------------------------------------------------------------------
def fig_metasurface():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    # grid of meta-atoms
    ax1.set_xlim(0, 8); ax1.set_ylim(0, 8); ax1.axis("off")
    ax1.set_title("16 x 16 = 256 meta-atoms", fontsize=11, color=SLATE, fontweight="bold")
    n = 8
    phase_colors = [BLUE, GREEN, ORANGE, PURPLE]
    rng = np.random.default_rng(3)
    cell = 6.0 / n
    x0, y0 = 1.0, 1.0
    for i in range(n):
        for j in range(n):
            c = phase_colors[rng.integers(0, 4)]
            r = Rectangle((x0 + j * cell, y0 + i * cell), cell * 0.9, cell * 0.9,
                          fc=c, ec="white", lw=1.0)
            ax1.add_patch(r)
    ax1.text(4.0, 0.4, "each atom = programmable phase shifter",
             ha="center", fontsize=9, color=INK, style="italic")

    # 2-bit phase states on unit circle
    ax2.set_xlim(-1.5, 1.5); ax2.set_ylim(-1.5, 1.5); ax2.axis("off")
    ax2.set_aspect("equal")
    ax2.set_title("2-bit phase states", fontsize=11, color=SLATE, fontweight="bold")
    circ = Circle((0, 0), 1.0, fill=False, ec=GREY, lw=1.4)
    ax2.add_patch(circ)
    states = [(0, "0"), (np.pi / 2, r"$\pi/2$"), (np.pi, r"$\pi$"), (3 * np.pi / 2, r"$3\pi/2$")]
    for ang, lbl in states:
        x, y = np.cos(ang), np.sin(ang)
        arrow(ax2, 0, 0, x, y, color=SOFT, lw=2.0, mut=12)
        ax2.plot(x, y, "o", color=SLATE, ms=8)
        ax2.text(1.25 * x, 1.25 * y, lbl, ha="center", va="center", fontsize=11, color=INK)
    ax2.text(0, -1.42, "4 discrete choices per atom", ha="center", fontsize=9,
             color=INK, style="italic")
    fig.tight_layout()
    save(fig, "fig_metasurface.png")


# ---------------------------------------------------------------------------
# Figure 5: multipath cancellation via zero-mean symbols
# ---------------------------------------------------------------------------
def fig_multipath():
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.0), sharey=True)
    t = np.linspace(0, 1, 400)
    # (a) baseband zero-mean symbol
    sig = np.sign(np.sin(2 * np.pi * 3 * t))
    axes[0].plot(t, sig, color=SOFT, lw=2)
    axes[0].axhline(0, color=GREY, lw=1)
    axes[0].set_title("(a) Symbol: zero mean\nover its period", fontsize=10, color=SLATE)

    # (b) environmental multipath: still zero mean
    env = 0.6 * np.sign(np.sin(2 * np.pi * 3 * t + 0.6)) + 0.3 * np.sign(np.sin(2 * np.pi * 3 * t + 1.8))
    axes[1].plot(t, env, color=ORANGE, lw=2)
    axes[1].axhline(0, color=GREY, lw=1)
    axes[1].set_title("(b) Env. multipath:\nsum = 0 (cancels)", fontsize=10, color="#b5651d")

    # (c) MTS path: weights vary within symbol -> non-zero mean
    mts = np.piecewise(t, [t < 0.25, (t >= 0.25) & (t < 0.5), (t >= 0.5) & (t < 0.75), t >= 0.75],
                       [0.9, 0.4, 0.7, 1.0])
    axes[2].plot(t, mts, color=GREEN, lw=2)
    axes[2].axhline(0, color=GREY, lw=1)
    axes[2].set_title("(c) MTS path: weights vary\nsum != 0 (survives)", fontsize=10, color="#3d7a4a")

    for ax in axes:
        ax.set_ylim(-1.4, 1.4)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_multipath.png")


# ---------------------------------------------------------------------------
# Figure 6: accuracy across datasets (from Table 1)
# ---------------------------------------------------------------------------
def fig_accuracy():
    datasets = ["MNIST", "Fashion", "Fruits", "AFHQ", "CelebA", "Widar3.0"]
    metaai = [89.77, 80.86, 85.05, 81.47, 75.00, 84.67]
    discrete = [72.05, 66.52, 68.77, 68.20, 57.47, 70.67]
    resnet = [99.62, 93.55, 99.82, 96.07, 90.91, 95.00]
    x = np.arange(len(datasets)); w = 0.26
    fig, ax = plt.subplots(figsize=(10.0, 3.8))
    ax.bar(x - w, resnet, w, label="ResNet-18 (digital, ref.)", color=GREY)
    ax.bar(x, metaai, w, label="MetaAI (prototype)", color=SOFT)
    ax.bar(x + w, discrete, w, label="DiscreteNN (baseline)", color=ORANGE)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 108)
    ax.set_xticks(x); ax.set_xticklabels(datasets)
    for xi, v in zip(x, metaai):
        ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=8.5, color=SLATE, fontweight="bold")
    ax.legend(loc="lower center", ncol=3, fontsize=8.5, frameon=False,
              bbox_to_anchor=(0.5, -0.28))
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_accuracy.png")


# ---------------------------------------------------------------------------
# Figure 7: accuracy vs number of meta-atoms (saturation)
# ---------------------------------------------------------------------------
def fig_metaatoms():
    n = np.array([1, 64, 256, 576, 1024])
    acc = np.array([18, 74, 90, 91, 91.5])
    wdd = np.array([0.05, 0.55, 0.95, 0.98, 0.99])
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    ax.plot(n, acc, "-o", color=SOFT, lw=2.2, ms=7, label="Recognition accuracy")
    ax.axvline(256, color=ORANGE, ls="--", lw=1.5)
    ax.text(256, 40, "  256 atoms:\n  best trade-off", fontsize=9, color="#b5651d")
    ax.set_xlabel("Number of meta-atoms")
    ax.set_ylabel("Accuracy (%)", color=SOFT)
    ax.set_ylim(0, 100)
    ax2 = ax.twinx()
    ax2.plot(n, wdd, "-s", color=GREEN, lw=2.0, ms=6, label="Weight distribution density")
    ax2.set_ylabel("WDD", color=GREEN)
    ax2.set_ylim(0, 1.05)
    for sp in ["top"]:
        ax.spines[sp].set_visible(False); ax2.spines[sp].set_visible(False)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    fig.tight_layout()
    save(fig, "fig_metaatoms.png")


# ---------------------------------------------------------------------------
# Figure 8: multi-sensor fusion improvement
# ---------------------------------------------------------------------------
def fig_multisensor():
    groups = ["Multi-PIE\n(3 camera views)", "RF-Sauron\n(3 antennas)", "USC-HAD\n(accel + gyro)"]
    one = [64.58, 70.0, 62.0]
    full = [89.58, 88.0, 89.06]
    x = np.arange(len(groups)); w = 0.32
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.bar(x - w / 2, one, w, label="Single sensor", color=GREY)
    ax.bar(x + w / 2, full, w, label="Fused sensors", color=SOFT)
    for xi, a, b in zip(x, one, full):
        ax.annotate(f"+{b - a:.1f}", xy=(xi, b + 1.5), ha="center", fontsize=9,
                    color="#3d7a4a", fontweight="bold")
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 108)
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=9)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_multisensor.png")


# ---------------------------------------------------------------------------
# Figure 9: training-to-deployment workflow
# ---------------------------------------------------------------------------
def fig_workflow():
    fig, ax = plt.subplots(figsize=(11.0, 2.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
    steps = [
        ("Train complex-\nvalued linear NN", BLUE),
        ("Get desired\nweights H_des", GREY),
        ("Solve for atom\nphases (Eqn. 7)", PURPLE),
        ("Load phases\nonto metasurface", GREEN),
        ("Compute in the\nair at runtime", ORANGE),
    ]
    w = 2.0; gap = 0.35; x = 0.3
    for i, (txt, c) in enumerate(steps):
        box(ax, x, 1.1, w, 1.8, txt, c, fs=9, bold=(i == 4))
        if i < len(steps) - 1:
            arrow(ax, x + w, 2.0, x + w + gap, 2.0, color=SOFT, lw=2.0)
        x += w + gap
    save(fig, "fig_workflow.png")


# ---------------------------------------------------------------------------
# Figure 10: benefits summary (energy + privacy)
# ---------------------------------------------------------------------------
def fig_benefits():
    fig, ax = plt.subplots(figsize=(10.5, 3.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    items = [
        ("IoT battery saved", "device does not run AI", BLUE),
        ("Privacy by design", "server sees results,\nnot raw data", GREEN),
        ("Single metasurface", "shared by many\nsensors", PURPLE),
        ("Standard links", "commodity IoT,\nno RF pre-coding", ORANGE),
    ]
    w = 2.6; gap = 0.35; x = 0.4
    for title_t, sub, c in items:
        box(ax, x, 1.0, w, 3.0, "", c, rounded=0.05)
        ax.text(x + w / 2, 3.2, title_t, ha="center", fontsize=10.5, color=INK, fontweight="bold")
        ax.text(x + w / 2, 1.9, sub, ha="center", fontsize=9, color=INK)
        x += w + gap
    save(fig, "fig_benefits.png")


if __name__ == "__main__":
    fig_paradigms()
    fig_sequential()
    fig_pipeline()
    fig_metasurface()
    fig_multipath()
    fig_accuracy()
    fig_metaatoms()
    fig_multisensor()
    fig_workflow()
    fig_benefits()
    print("all MetaAI figures written to", OUT)
