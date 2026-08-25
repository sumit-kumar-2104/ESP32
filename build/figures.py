"""Generate figures and rendered equations for the wireless wave-perception document."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(OUT, exist_ok=True)

# Light, presentable palette
INK = "#33373d"
SOFT = "#5b7fa6"      # slate blue
BLUE = "#9ecae1"
GREEN = "#a8d5a2"
ORANGE = "#f4b183"
PURPLE = "#c4b7e0"
GREY = "#d9dce1"
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
# Equation rendering via matplotlib mathtext (clean typeset math, no LaTeX needed)
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
    "eq_helmholtz.png": r"$\nabla^{2}U(\mathbf{r}) + k^{2}\,U(\mathbf{r}) = 0,\qquad k = \dfrac{2\pi}{\lambda} = \dfrac{\omega}{c}$",
    "eq_rs.png": r"$U(x,y,z)=\iint U_{0}(x',y')\,h(x-x',\,y-y';\,z)\,dx'dy'$",
    "eq_rs_kernel.png": r"$h(x,y;z)=\dfrac{z}{j\lambda}\,\dfrac{e^{\,jkr}}{r^{2}},\qquad r=\sqrt{x^{2}+y^{2}+z^{2}}$",
    "eq_angular.png": r"$U(\cdot,z)=\mathcal{F}^{-1}\!\left\{\,\mathcal{F}\{U_{0}\}\;e^{\,jz\sqrt{k^{2}-k_x^{2}-k_y^{2}}}\,\right\}$",
    "eq_layer.png": r"$U_{l+1}=\mathcal{P}_{d}\left\{\,t_{l}(x,y)\,U_{l}(x,y)\,\right\},\qquad t_{l}=a_{l}\,e^{\,j\phi_{l}}$",
    "eq_output.png": r"$\hat{y}= \left|\,U_{L}(x,y)\,\right|^{2}$",
    "eq_scale.png": r"$x\!\to\! s x,\ \ z\!\to\! s z,\ \ \lambda\!\to\! s\lambda \quad\Rightarrow\quad k\,r \ \ \mathrm{invariant}$",
    "eq_res.png": r"$\delta_{\min}\approx\dfrac{\lambda}{2\,\mathrm{NA}}$",
    "eq_channel.png": r"$\mathbf{y}=\mathbf{H}\,\mathbf{x}+\mathbf{n}$",
    "eq_csi.png": r"$H_{k}=\sum_{p=1}^{P} a_{p}\,e^{-j2\pi f_{k}\tau_{p}}$",
    "eq_csi_dyn.png": r"$H(t)=H_{\mathrm{static}}+\sum_{m}\alpha_{m}(t)\,e^{-j2\pi d_{m}(t)/\lambda}$",
    "eq_shannon.png": r"$N_{\mathrm{dof}}\approx\dfrac{A_{T}\,A_{R}}{(\lambda d)^{2}}$",
    "eq_range.png": r"$\Delta R=\dfrac{c}{2B}$",
    "eq_angle.png": r"$\Delta\theta\approx\dfrac{\lambda}{D}$",
    "eq_cs.png": r"$M\ \geq\ C\,S\,\log\!\left(N/S\right)$",
    "eq_fusion.png": r"$p(\mathbf{s}\mid \mathbf{z}_{\mathrm{rf}},\mathbf{z}_{\mathrm{cam}})\ \propto\ p(\mathbf{z}_{\mathrm{rf}}\mid \mathbf{s})\,p(\mathbf{z}_{\mathrm{cam}}\mid \mathbf{s})\,p(\mathbf{s})$",
}
for fname, tex in equations.items():
    render_eq(tex, fname)


# ---------------------------------------------------------------------------
# Figure 1: one wave equation across the spectrum
# ---------------------------------------------------------------------------
def fig_spectrum():
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    ax.set_xscale("log")
    ax.set_xlim(1e-7, 1e0)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color=GREY, lw=8, solid_capstyle="round", zorder=1)
    points = [
        (0.5e-6, "Visible light\n~0.5 um", SOFT),
        (5e-3, "mmWave 60 GHz\n~5 mm", ORANGE),
        (6e-2, "WiFi 5 GHz\n~6 cm", GREEN),
    ]
    for x, lbl, c in points:
        ax.plot([x], [0.5], "o", ms=16, color=c, zorder=3, markeredgecolor="white", markeredgewidth=1.5)
        ax.annotate(lbl, (x, 0.5), (x, 0.85), ha="center", va="center", fontsize=9.5,
                    color=INK, arrowprops=dict(arrowstyle="-", color="#aeb3ba"))
    ax.text(0.5, 0.12, r"All obey the same Helmholtz equation:  $\nabla^{2}U+k^{2}U=0$  with  $k=2\pi/\lambda$",
            transform=ax.transAxes, ha="center", fontsize=11, color=SOFT, style="italic")
    ax.set_yticks([])
    ax.set_xlabel("Wavelength  (m, log scale)")
    for s in ["top", "left", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_title("One physics, many wavelengths: optics and wireless are the same wave", color=INK, fontsize=12, pad=10)
    save(fig, "fig_spectrum.png")


# ---------------------------------------------------------------------------
# Figure 2: optical diffractive network vs scaled RF metasurface network
# ---------------------------------------------------------------------------
def _stack(ax, x0, scale, color, title, layer_label):
    # input plane
    planes_x = [x0 + scale * i for i in range(0, 5)]
    h = 1.6 * 1
    for i, px in enumerate(planes_x):
        if i == 0:
            c = "#f0f0f0"; lbl = "input\nfield"
        elif i == len(planes_x) - 1:
            c = "#eef4ee"; lbl = "detector\n$|U|^2$"
        else:
            c = color; lbl = layer_label
        rect = FancyBboxPatch((px - 0.06 * scale, 0.0), 0.12 * scale, h,
                              boxstyle="round,pad=0.01", linewidth=1.0,
                              edgecolor="#9aa0a8", facecolor=c)
        ax.add_patch(rect)
        ax.text(px, -0.28, lbl, ha="center", va="top", fontsize=8.0, color=INK)
    # connecting wave arrows
    for a, b in zip(planes_x[:-1], planes_x[1:]):
        ax.annotate("", (b - 0.07 * scale, h / 2), (a + 0.07 * scale, h / 2),
                    arrowprops=dict(arrowstyle="-|>", color="#b6bcc4", lw=1.2))
    ax.text((planes_x[0] + planes_x[-1]) / 2, h + 0.25, title, ha="center",
            fontsize=11, color=SOFT, weight="bold")


def fig_diffractive():
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    _stack(ax, 0.4, 0.72, BLUE, "", "phase\nmask")
    _stack(ax, 4.7, 1.02, ORANGE, "", "meta-\nsurface")
    ax.text(1.84, 3.0, "Optical diffractive network\n($\\lambda\\sim0.5\\,\\mu$m, cm-scale)", ha="center",
            fontsize=10.5, color=INK)
    ax.text(7.25, 3.0, "RF metasurface network\n($\\lambda\\sim$cm, m-scale)", ha="center",
            fontsize=10.5, color=INK)
    ax.annotate("identical topology,\ngeometry scaled by  $s=\\lambda_{RF}/\\lambda_{opt}$",
                (4.0, 1.4), (4.0, 2.5), ha="center", fontsize=9.5, color=PURPLE,
                arrowprops=dict(arrowstyle="-", color=PURPLE))
    ax.set_xlim(0, 9.6)
    ax.set_ylim(-0.7, 3.6)
    ax.axis("off")
    ax.set_title("The same trainable wave-propagation model at two wavelengths", fontsize=12, pad=6)
    save(fig, "fig_diffractive.png")


# ---------------------------------------------------------------------------
# Figure 3: diffraction-limited resolution vs wavelength
# ---------------------------------------------------------------------------
def fig_resolution():
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    lam = np.logspace(-7, -1, 200)  # m
    NA = 0.5
    delta = lam / (2 * NA)
    ax.loglog(lam * 1e2, delta * 1e2, color=SOFT, lw=2.2)
    marks = [(0.5e-6, "Visible", SOFT), (5e-3, "mmWave 60 GHz", ORANGE), (6e-2, "WiFi 5 GHz", GREEN)]
    for x, lbl, c in marks:
        d = x / (2 * NA)
        ax.plot(x * 1e2, d * 1e2, "o", ms=11, color=c, markeredgecolor="white", markeredgewidth=1.3, zorder=5)
        ax.annotate(lbl, (x * 1e2, d * 1e2), (x * 1e2 * 1.4, d * 1e2 * 0.35),
                    fontsize=9, color=INK)
    ax.set_xlabel("Wavelength  (cm)")
    ax.set_ylabel("Smallest resolvable feature  (cm)")
    ax.set_title(r"Resolution scales with wavelength:  $\delta_{\min}\approx\lambda/(2\,\mathrm{NA})$", fontsize=11.5)
    ax.grid(True, which="both", color="#eef0f3", lw=0.8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    save(fig, "fig_resolution.png")


# ---------------------------------------------------------------------------
# Figure 4: meta-perception fusion architecture
# ---------------------------------------------------------------------------
def _box(ax, x, y, w, h, text, color, fs=9.5):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", linewidth=1.1,
                       edgecolor="#9aa0a8", facecolor=color)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)


def _arrow(ax, p1, p2, label=None, color="#aeb3ba"):
    ax.annotate("", p2, p1, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6))
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.12, label, ha="center", fontsize=8, color=SOFT)


def fig_fusion():
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    _box(ax, 0.2, 2.6, 1.9, 0.9, "Wireless sensor\n(WiFi CSI / radar)\nALWAYS ON", BLUE)
    _box(ax, 0.2, 0.5, 1.9, 0.9, "Camera\n(RGB / depth)\nON DEMAND", GREEN)
    _box(ax, 2.9, 2.6, 1.7, 0.9, "RF perception\nmodel", "#eef4f8")
    _box(ax, 2.9, 0.5, 1.7, 0.9, "Vision\nmodel", "#eef6ee")
    _box(ax, 5.3, 1.55, 1.7, 0.9, "Multi-modal\nfusion", PURPLE)
    _box(ax, 7.5, 1.55, 1.5, 0.9, "Decision /\napplication", ORANGE)
    _box(ax, 2.9, 1.62, 1.7, 0.7, "trigger / wake", "#fbe9d6", fs=8.5)
    _arrow(ax, (2.1, 3.05), (2.9, 3.05))
    _arrow(ax, (2.1, 0.95), (2.9, 0.95))
    _arrow(ax, (4.6, 3.05), (5.3, 2.25))
    _arrow(ax, (4.6, 0.95), (5.3, 1.75))
    _arrow(ax, (3.75, 2.6), (3.75, 2.32), color=ORANGE)
    _arrow(ax, (3.75, 1.62), (3.75, 1.4), color=ORANGE)
    _arrow(ax, (7.0, 2.0), (7.5, 2.0))
    ax.text(3.75, 2.45, "", ha="center")
    ax.set_xlim(0, 9.2)
    ax.set_ylim(0, 4.0)
    ax.axis("off")
    ax.set_title("Meta-perception: wireless runs continuously, vision wakes on confident events",
                 fontsize=11.5, pad=6)
    save(fig, "fig_fusion.png")


# ---------------------------------------------------------------------------
# Figure 5: event camera vs continuous wireless monitoring
# ---------------------------------------------------------------------------
def fig_event_vs_wireless():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.8, 4.4), sharex=True)
    t = np.linspace(0, 10, 1000)
    motion = ((t > 2) & (t < 3.2)) | ((t > 6.5) & (t < 7.3))
    # Event camera: spikes only during motion
    rng = np.random.default_rng(3)
    ev = np.zeros_like(t)
    ev[motion] = rng.uniform(0.2, 1.0, motion.sum())
    markers = t[motion][::12]
    ax1.vlines(markers, 0, rng.uniform(0.4, 1.0, len(markers)), color=SOFT, lw=1.4)
    ax1.axhline(0, color="#cfd3d9", lw=1)
    ax1.set_ylim(-0.1, 1.15)
    ax1.set_yticks([])
    ax1.set_title("Event camera: emits data only when the scene changes (silent on a still person)",
                  fontsize=10.5, color=INK, loc="left")
    ax1.text(4.6, 0.85, "static person -> no events (blind)", fontsize=9, color="#b06a3b")
    for s in ["top", "right", "left"]:
        ax1.spines[s].set_visible(False)
    # Wireless: continuous signal; background shift for static presence + perturbations for motion
    base = 0.45 + 0.0 * t
    base[t > 1.0] = 0.62  # person enters and stays -> static multipath shifts
    sig = base.copy()
    sig[motion] += 0.18 * np.sin(2 * np.pi * 6 * t[motion]) * np.hanning(1000)[motion]
    sig += 0.012 * rng.standard_normal(len(t))
    ax2.plot(t, sig, color=GREEN, lw=1.6)
    ax2.axhline(0.45, color="#cfd3d9", lw=1.2, ls="--")
    ax2.text(0.1, 0.39, "empty-room reference", fontsize=8.5, color="#7a7f86")
    ax2.annotate("static presence\n(background shift)", (1.6, 0.62), (3.0, 0.30),
                 fontsize=9, color=GREEN, arrowprops=dict(arrowstyle="-|>", color=GREEN))
    ax2.annotate("motion (Doppler)", (2.6, 0.78), (4.2, 0.86),
                 fontsize=9, color=SOFT, arrowprops=dict(arrowstyle="-|>", color=SOFT))
    ax2.set_ylim(0.25, 0.95)
    ax2.set_yticks([])
    ax2.set_xlabel("time (s)")
    ax2.set_title("Active wireless: always illuminated, so it sees a still person and motion alike",
                  fontsize=10.5, color=INK, loc="left")
    for s in ["top", "right", "left"]:
        ax2.spines[s].set_visible(False)
    save(fig, "fig_event_vs_wireless.png")


# ---------------------------------------------------------------------------
# Figure 6: data requirements (range resolution vs bandwidth, channels vs aperture)
# ---------------------------------------------------------------------------
def fig_data_requirements():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.9, 3.4))
    B = np.linspace(20e6, 4e9, 200)
    dR = 3e8 / (2 * B)
    ax1.plot(B / 1e9, dR * 100, color=SOFT, lw=2.2)
    for bw, lbl, c in [(20e6, "WiFi 20 MHz", GREEN), (1.6e9, "UWB 1.6 GHz", ORANGE), (4e9, "mmWave 4 GHz", PURPLE)]:
        ax1.plot(bw / 1e9, (3e8 / (2 * bw)) * 100, "o", ms=9, color=c, markeredgecolor="white")
        ax1.annotate(lbl, (bw / 1e9, (3e8 / (2 * bw)) * 100), fontsize=8, color=INK,
                     xytext=(bw / 1e9 + 0.2, (3e8 / (2 * bw)) * 100 + 30))
    ax1.set_xlabel("Bandwidth  B  (GHz)")
    ax1.set_ylabel("Range resolution  $\\Delta R$  (cm)")
    ax1.set_title(r"$\Delta R = c/2B$", fontsize=10.5)
    ax1.set_ylim(0, 800)
    for s in ["top", "right"]:
        ax1.spines[s].set_visible(False)
    ax1.grid(True, color="#eef0f3")

    D = np.linspace(0.1, 2.0, 200)  # aperture size (m), square apertures A=D^2
    lam = 0.06
    d = 4.0
    N = (D**2 * D**2) / (lam * d) ** 2
    ax2.plot(D * 100, N, color=ORANGE, lw=2.2)
    ax2.set_xlabel("Aperture size  D  (cm)")
    ax2.set_ylabel("Independent channels  $N_{dof}$")
    ax2.set_title(r"$N_{dof}\approx A_T A_R/(\lambda d)^2$", fontsize=10.5)
    for s in ["top", "right"]:
        ax2.spines[s].set_visible(False)
    ax2.grid(True, color="#eef0f3")
    fig.suptitle("Sizing the measurement: what data the wireless front-end must capture",
                 fontsize=11.5, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig_data_requirements.png")


if __name__ == "__main__":
    fig_spectrum()
    fig_diffractive()
    fig_resolution()
    fig_fusion()
    fig_event_vs_wireless()
    fig_data_requirements()
    print("ALL FIGURES DONE")
