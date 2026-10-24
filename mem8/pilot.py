"""Pilot verifications for the Mem8 monograph. Deterministic under fixed seeds."""
import numpy as np

rng = np.random.default_rng(8)


def S(psi, q):
    """Listopad resonance score, Eq. (2)-(3) of arXiv:2509.09691, for complex vectors."""
    e1 = np.vdot(psi, psi).real
    e2 = np.vdot(q, q).real
    if e1 + e2 == 0:
        return 0.0
    r = 2 * np.sqrt(e1 * e2) / (e1 + e2)
    num = np.sum(np.abs(psi + q) ** 2)
    return 0.5 * num / (e1 + e2) * r


def S_closed(psi, q):
    """Closed form S = (rho/2)(1 + rho c)."""
    a = np.linalg.norm(psi)
    b = np.linalg.norm(q)
    rho = 2 * a * b / (a * a + b * b)
    c = np.vdot(psi, q).real / (a * b)
    return 0.5 * rho * (1 + rho * c)


def realify(z):
    return np.concatenate([z.real, z.imag])


out = {}

# (a) closed form versus definition, unequal energies
n = 64
err = 0.0
for _ in range(20000):
    psi = rng.normal(size=n) + 1j * rng.normal(size=n)
    q = (rng.normal(size=n) + 1j * rng.normal(size=n)) * rng.uniform(0.1, 5)
    err = max(err, abs(S(psi, q) - S_closed(psi, q)))
out["closed_form_max_abs_err"] = err

# (b) normalized reduction S = (1 + Re<psi,q>)/2
err = 0.0
for _ in range(20000):
    psi = rng.normal(size=n) + 1j * rng.normal(size=n)
    q = rng.normal(size=n) + 1j * rng.normal(size=n)
    psi /= np.linalg.norm(psi)
    q /= np.linalg.norm(q)
    err = max(err, abs(S(psi, q) - 0.5 * (1 + np.vdot(psi, q).real)))
out["normalized_reduction_max_abs_err"] = err

# (c) realification: same score from the real inner product on R^{2n}
err = 0.0
for _ in range(20000):
    psi = rng.normal(size=n) + 1j * rng.normal(size=n)
    q = rng.normal(size=n) + 1j * rng.normal(size=n)
    x, y = realify(psi), realify(q)
    a, b = np.linalg.norm(x), np.linalg.norm(y)
    rho = 2 * a * b / (a * a + b * b)
    c = (x @ y) / (a * b)
    err = max(err, abs(0.5 * rho * (1 + rho * c) - S(psi, q)))
out["realification_max_abs_err"] = err

# (d) range of S over random draws, including the minimum 1/8 at rho=1/2, c=-1
vals = []
for _ in range(200000):
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    q = -psi * rng.uniform(0.0, 1.0)
    vals.append(S(psi, q))
out["min_S_antiphase_scaled"] = float(np.min(vals))
out["theory_min_antiphase"] = 1 / 8

# (e) false activation bound: P(Re<m,q> > a) <= (1-a^2)^(n-1) for unit vectors in C^n
def unit(n):
    z = rng.normal(size=n) + 1j * rng.normal(size=n)
    return z / np.linalg.norm(z)

rows = []
for n_ in (16, 32, 64):
    for a in (0.2, 0.3, 0.4):
        trials = 400000
        m = unit(n_)
        cnt = 0
        for _ in range(trials // 1000):
            Z = rng.normal(size=(1000, n_)) + 1j * rng.normal(size=(1000, n_))
            Z /= np.linalg.norm(Z, axis=1, keepdims=True)
            cnt += int(np.sum((Z @ np.conj(m)).real > a))
        emp = cnt / (trials // 1000 * 1000)
        bound = (1 - a * a) ** (n_ - 1)
        rows.append((n_, a, emp, bound))
out["false_activation"] = rows

# (f) baseline recurrence: with scores independent of r and theta frozen, r_k converges to sum w_i m_i
#     at geometric rate (1-eta)^k exactly.
P, n_, eta, beta = 20, 32, 0.3, 12.0
M = np.array([unit(n_) for _ in range(P)])
cue = M[3] * 0.0
cue = M[3].copy()
cue[: n_ // 2] = 0
cue = cue / np.linalg.norm(cue)
s = np.array([0.5 * (1 + np.vdot(m, cue).real) for m in M])
w = np.exp(beta * s)
w /= w.sum()
target = (w[:, None] * M).sum(0)
r = np.zeros(n_, dtype=complex)
ratios = []
prev = np.linalg.norm(r - target)
for k in range(12):
    r = (1 - eta) * r + eta * target
    cur = np.linalg.norm(r - target)
    ratios.append(cur / prev)
    prev = cur
out["baseline_contraction_ratios"] = ratios
out["baseline_expected_ratio"] = 1 - eta
out["baseline_top_weight_index"] = int(np.argmax(w))
out["baseline_top_weight"] = float(w.max())
out["baseline_cosine_to_true_trace"] = float(np.vdot(M[3], r).real / np.linalg.norm(r))

# (g) Kuramoto gradient structure: identical frequencies, symmetric K, V nonincreasing under small step
N = 30
K = rng.uniform(0.0, 1.0, size=(N, N))
K = (K + K.T) / 2
np.fill_diagonal(K, 0)
th = rng.uniform(0, 2 * np.pi, size=N)
dt = 0.9 / np.max(np.abs(K).sum(1))


def V(th):
    D = th[None, :] - th[:, None]
    return -0.5 * np.sum(K * np.cos(D))


vs = [V(th)]
for _ in range(400):
    D = th[None, :] - th[:, None]
    th = th + dt * np.sum(K * np.sin(D), axis=1)
    vs.append(V(th))
vs = np.array(vs)
out["kuramoto_max_increase_of_V"] = float(np.max(np.diff(vs)))
out["kuramoto_V_start_end"] = (float(vs[0]), float(vs[-1]))
R = abs(np.mean(np.exp(1j * th)))
out["kuramoto_final_order_parameter"] = float(R)

for k, v in out.items():
    print(k, v)
