"""Reference checks for the edge-case fixtures of Appendix H.
Run with: python3 -I tools/fixtures_check.py
Each check prints the value quoted in the specification and asserts it."""
import math

def dist(p, q): return math.hypot(p[0]-q[0], p[1]-q[1])
def mean_d(A, B): return sum(dist(a, b) for a, b in zip(A, B)) / len(A)

# ---- symmetric DTW (Chapter 18, Listing C.3) -------------------------------
def dtw(a, b, w=None):
    n, m = len(a), len(b)
    w = max(abs(n-m), w if w is not None else max(n, m))
    INF = float('inf')
    D = [[INF]*(m+1) for _ in range(n+1)]
    D[0][0] = 0.0
    for i in range(1, n+1):
        for j in range(max(1, i-w), min(m, i+w)+1):
            c = dist(a[i-1], b[j-1])
            D[i][j] = min(D[i-1][j]+c, D[i][j-1]+c, D[i-1][j-1]+2*c)
    return D[n][m] / (n+m)

a = [(0,0),(2,0)]; b = [(0,0),(1,0),(2,0)]
v = dtw(a, b); print("dtw hand example", v); assert abs(v-0.2) < 1e-12
# diagonal path equals the pointwise distance; dtw never exceeds it
P = [(0,0),(1,0),(2,1),(3,1)]; Q = [(0,0.2),(1,0.1),(2,1.2),(3,0.7)]
assert dtw(P, Q) <= mean_d(P, Q) + 1e-12
print("dtw <= pointwise:", dtw(P, Q), mean_d(P, Q))

# ---- decision rule (Chapter 21, Listing C.4) -------------------------------
def decide(dist_by_g, prior, sigma=0.1, lam=1.0, theta=1.0,
           tau=0.8, mu=0.3, pmin=0.15, dmax=0.3):
    if not dist_by_g: return ("Rejected", "no_candidates")
    names = list(dist_by_g)           # library order
    logp = {g: -dist_by_g[g]**2/(2*sigma**2) + lam*math.log(prior[g]) for g in names}
    mx = max(logp.values())
    ex = {g: math.exp((logp[g]-mx)/theta) for g in names}
    Z = sum(ex.values()); p = {g: ex[g]/Z for g in names}
    gstar = sorted(names, key=lambda g: (-round(p[g], 12), dist_by_g[g], names.index(g)))[0]
    p2 = 0.0 if len(names) == 1 else max(p[g] for g in names if g != gstar)
    gnn = min(names, key=lambda g: dist_by_g[g])
    if dist_by_g[gstar] > dmax: return ("Rejected", "too_far", p)
    if p[gstar] >= tau and p[gstar]-p2 >= mu:
        return ("Accepted", gstar, round(p[gstar], 4), "override" if gstar != gnn else "", p)
    return ("Ambiguous", sorted(g for g in names if p[g] >= pmin), p)

print("empty        ", decide({}, {}))
print("single near  ", decide({"A": 0.10}, {"A": 1.0})[:3])
print("single far   ", decide({"A": 0.40}, {"A": 1.0})[:2])
print("tie          ", decide({"A": 0.10, "B": 0.10}, {"A": .5, "B": .5})[:2])
r = decide({"A": 0.10, "B": 0.12}, {"A": 0.1, "B": 0.9})
print("override     ", r[:4], {k: round(x, 4) for k, x in r[4].items()})
assert r[0] == "Accepted" and r[1] == "B" and r[3] == "override"
r = decide({"A": 0.10, "B": 0.12}, {"A": 0.5, "B": 0.5})
print("uniform prior", r[:2], {k: round(x, 4) for k, x in r[2].items()})

# ---- invariance asymmetry and library validity ------------------------------
ts = [-0.5, -0.25, 0.0, 0.25, 0.5]
H = [(x, 0.0) for x in ts]; V = [(0.0, y) for y in ts]
c_v_of_h = mean_d(H, V)      # candidate line.v (no rotation), stroke = horizontal
print("c_v(h,v)", round(c_v_of_h, 4), " c_h(v,h) = 0 (rotation allowed)")
assert abs(c_v_of_h-0.424264) < 1e-5

# ---- bounded update (Chapter 23, Listing C.6) --------------------------------
T0 = list(V); Ghat = [(x+0.2, y) for x, y in V]; eta, rho = 0.5, 0.05
T = list(T0)
T1 = [((1-eta)*t[0]+eta*g[0], (1-eta)*t[1]+eta*g[1]) for t, g in zip(T, Ghat)]
r = mean_d(T1, T0); print("drift before retraction", r)
T1 = [(t0[0]+(rho/r)*(t1[0]-t0[0]), t0[1]+(rho/r)*(t1[1]-t0[1])) for t0, t1 in zip(T0, T1)]
print("drift after retraction ", mean_d(T1, T0)); assert abs(mean_d(T1, T0)-rho) < 1e-12
