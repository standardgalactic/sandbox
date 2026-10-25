import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.linewidth":0.7,
    "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
A = np.genfromtxt("/home/claude/build/data/composite_sasm.csv", delimiter=",", skip_header=1)
age, d18, lo, hi, n, _,_,_, sasm, r, p = A.T
BLUE, ORANGE, GREY, INK = "#1f5f8b", "#c4651b", "#7a7a7a", "#222222"
terms = {"TI":(11,18), "TII":(128,136), "TIII":(242,250), "TIV":(335,343), "TV":(424,432)}

valid = ~np.isnan(r)
print("valid windows", valid.sum())
print("r<0 frac", np.mean(r[valid]<0).round(3))
print("r<0 & p<.05 frac", np.mean((r[valid]<0)&(p[valid]<0.05)).round(3))
print("r>0 & p<.05 frac", np.mean((r[valid]>0)&(p[valid]<0.05)).round(3))
print("r>0 frac", np.mean(r[valid]>0).round(3))

fig, axs = plt.subplots(3, 1, figsize=(6.4, 6.2), sharex=True)
def shade(ax):
    for k,(a,b) in terms.items():
        ax.axvspan(a, b, color="#e1dccf", alpha=0.75, lw=0)
for ax in axs: shade(ax)
ax = axs[0]
ax.fill_between(age, lo, hi, color=BLUE, alpha=0.25, lw=0)
ax.plot(age, d18, color=BLUE, lw=0.8)
ax.set_ylabel("Andean composite\n"+r"$\delta^{18}$O (‰, VPDB)")
ax.text(446, ax.get_ylim()[1], "drier $\\uparrow$", ha="right", va="top", fontsize=8)
for k,(a,b) in terms.items():
    ax.text((a+b)/2, ax.get_ylim()[0]+0.15, k, ha="center", va="bottom", fontsize=7.5)
ax = axs[1]
ax.plot(age, sasm, color=INK, lw=0.8)
ax.axhline(0, color=INK, lw=0.5)
ax.set_ylabel("regional SASM index\n(standardized, dry $+$)")
ax.set_ylim(-3.4, 3.6)
ax = axs[2]
ax.axhline(0, color=INK, lw=0.5)
rr = np.where(valid, r, np.nan)
ax.fill_between(age, 0, rr, where=(rr>0), color=ORANGE, alpha=0.55, lw=0)
ax.fill_between(age, 0, rr, where=(rr<0), color=BLUE, alpha=0.40, lw=0)
ax.plot(age, rr, color=INK, lw=0.6)
sig = valid & (p < 0.05) & (r > 0)
ax.plot(age[sig], np.full(sig.sum(), 1.04), "|", color=ORANGE, ms=5, mew=1.0)
ax.text(2, 1.04, r"$p<0.05$, $r>0$", fontsize=7, va="center", color=ORANGE)
ax.set_ylim(-1.05, 1.15)
ax.set_ylabel("rolling correlation $r$\nwith Asian record")
ax.set_xlabel("age (ka BP)")
ax.set_xlim(0, 450)
fig.align_ylabels(axs)
fig.tight_layout(h_pad=0.5)
fig.savefig("/home/claude/build/figures/fig4_record.pdf")
print("saved")
