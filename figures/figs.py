import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "figure.dpi": 150,
})
out = "/home/claude/build/figures/"
BLUE, ORANGE, GREY, INK = "#1f5f8b", "#c4651b", "#7a7a7a", "#222222"

# ---------- Figure 1: idealized orbital dipole ----------
t = np.linspace(0, 100, 1000)           # kyr
I = np.cos(2*np.pi*t/23.0)
fig, ax = plt.subplots(figsize=(6.2, 2.7))
ax.plot(t, I, color=GREY, lw=1.0, ls=(0, (4, 2)), label=r"insolation contrast $I(t)$")
ax.plot(t, 0.8*I, color=ORANGE, lw=1.6, label=r"northern monsoon $M_N=a_N I$")
ax.plot(t, -0.8*I, color=BLUE, lw=1.6, label=r"southern monsoon $M_S=-a_S I$")
ax.axhline(0, color=INK, lw=0.5)
ax.set_xlabel("time (kyr, arbitrary origin)")
ax.set_ylabel("standardized units")
ax.set_xlim(0, 100); ax.set_ylim(-1.35, 1.25)
ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7.5, handlelength=2.2)
ax.text(99, -1.28, "schematic of eq. (1) with $a_N=a_S=0.8$; not data", ha="right", fontsize=7, color=GREY)
fig.tight_layout()
fig.savefig(out + "fig1_orbital_dipole.pdf")
plt.close(fig)

# ---------- Figure 2: common and difference modes ----------
dt = 0.05
tt = np.arange(0, 100, dt)
Iforc = np.cos(2*np.pi*tt/23.0)
lamD, a = 1.0, 1.0
lamC = 0.6
f = np.zeros_like(tt)
pulse = (tt > 52) & (tt < 58)
f[pulse] = -1.6                          # common-mode forcing (drying)
D = np.zeros_like(tt); C = np.zeros_like(tt)
for i in range(1, len(tt)):
    D[i] = D[i-1] + dt*(-lamD*D[i-1] + a*Iforc[i-1])
    C[i] = C[i-1] + dt*(-lamC*C[i-1] + f[i-1])
xN, xS = C + D, C - D

fig, axs = plt.subplots(3, 1, figsize=(6.2, 5.6), sharex=True,
                        gridspec_kw={"height_ratios": [1.15, 1, 1]})
ax = axs[0]
ax.plot(tt, xN, color=ORANGE, lw=1.4, label=r"$x_N=C+D$")
ax.plot(tt, xS, color=BLUE, lw=1.4, label=r"$x_S=C-D$")
ax.axhline(0, color=INK, lw=0.5)
ax.axvspan(52, 76, color="#e8e2d4", alpha=0.7, lw=0)
ax.set_ylabel("rainfall anomaly\n(wet $+$, dry $-$)")
ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=8)
ax.text(64, 1.55, "both dry", ha="center", fontsize=8, color=INK)
ax = axs[1]
ax.plot(tt, D, color=INK, lw=1.4)
ax.axhline(0, color=INK, lw=0.5)
ax.axvspan(52, 76, color="#e8e2d4", alpha=0.7, lw=0)
ax.set_ylabel(r"difference mode $D$")
ax.text(64, -1.35, "dipole continues", ha="center", fontsize=8)
ax.set_ylim(-1.7, 1.4)
ax = axs[2]
ax.plot(tt, C, color=GREY, lw=1.6)
ax.axhline(0, color=INK, lw=0.5)
ax.axhline(-1.0, color=INK, lw=0.6, ls=(0, (3, 2)))
ax.text(1, -1.12, "detection threshold (schematic)", fontsize=7, color=INK, va="top")
ax.axvspan(52, 76, color="#e8e2d4", alpha=0.7, lw=0)
ax.set_ylabel(r"common mode $C$")
ax.set_xlabel("time (kyr, arbitrary origin)")
ax.set_xlim(0, 100)
fig.align_ylabels(axs)
fig.tight_layout(h_pad=0.6)
fig.savefig(out + "fig2_common_difference.pdf")
plt.close(fig)

# ---------- Figure 3: bifurcation structure ----------
beta = 1.0
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.9))
# (a) pitchfork, h = 0
ax = axs[0]
mu_neg = np.linspace(-1, 0, 200)
mu_pos = np.linspace(0, 1, 200)
ax.plot(mu_neg, 0*mu_neg, color=INK, lw=1.6)
ax.plot(mu_pos, 0*mu_pos, color=INK, lw=1.0, ls=(0, (3, 2)))
ax.plot(mu_pos, np.sqrt(mu_pos/beta), color=INK, lw=1.6)
ax.plot(mu_pos, -np.sqrt(mu_pos/beta), color=INK, lw=1.6)
ax.set_xlabel(r"control parameter $\mu(\mathbf{B})$")
ax.set_ylabel(r"equilibrium $C^*$")
ax.set_title(r"(a) unforced, $h=0$", fontsize=9, loc="left")
ax.text(-0.95, 0.12, "one stable\nstate", fontsize=7.5)
ax.text(0.30, 1.05, "two stable states", fontsize=7.5)
ax.text(0.30, 0.10, "unstable", fontsize=7, rotation=0)
ax.set_xlim(-1, 1); ax.set_ylim(-1.3, 1.3)
# (b) fold, mu = 1
ax = axs[1]
mu = 1.0
h = np.linspace(-0.7, 0.7, 4000)
Cs = np.linspace(-1.6, 1.6, 8001)
hh = -(mu*Cs - beta*Cs**3)              # h such that mu C - beta C^3 + h = 0
stable = (mu - 3*beta*Cs**2) < 0
for sel, ls, lw in [(stable, "-", 1.6), (~stable, (0, (3, 2)), 1.0)]:
    x = np.where(sel, hh, np.nan); y = np.where(sel, Cs, np.nan)
    # split into contiguous runs
    idx = np.where(~np.isnan(x))[0]
    runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    for r in runs:
        if len(r) > 1: ax.plot(x[r], y[r], color=INK, lw=lw, ls=ls)
hc = 2*mu**1.5/(3*np.sqrt(3*beta))
for s in (-1, 1):
    ax.axvline(s*hc, color=GREY, lw=0.5, ls=":")
ax.set_xlabel(r"perturbation $h$")
ax.set_ylabel(r"equilibrium $C^*$")
ax.set_title(r"(b) forced, $\mu=1$", fontsize=9, loc="left")
ax.text(hc+0.02, -1.45, r"$|h|=\dfrac{2\mu^{3/2}}{3\sqrt{3\beta}}$", fontsize=8, va="bottom")
ax.set_xlim(-0.7, 0.7); ax.set_ylim(-1.6, 1.6)
fig.tight_layout(w_pad=1.5)
fig.savefig(out + "fig3_bifurcation.pdf")
plt.close(fig)
print("done")
