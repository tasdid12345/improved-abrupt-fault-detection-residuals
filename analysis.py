"""Abrupt-fault detection on a simulated mass-spring-damper system.
Baseline: ARX one-step-ahead residual with causal smoothing and a fault-free threshold.
Improved: class-balanced LightGBM on sliding-window residual/input/output features.
Seed 42 everywhere (improved model also run with 43, 44). Test set used once, at the end."""
import json, os, sys, time
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, ParameterGrid, train_test_split

SEED, SEEDS = 42, [42, 43, 44]
N_EP, T, DT, W, SMOOTH = 240, 500, 0.1, 20, 10  # episodes, samples, time step, window, baseline smoothing
t_start = time.time()
rng = np.random.default_rng(SEED)


def simulate(rng, faulty):
    """Euler simulation of m x'' + c x' + k x = u + noise. Faulty episodes change k or c abruptly
    at a random sample tf; fault-free episodes use tf = T (never reached)."""
    k, c = 4.0, 0.6
    tf = int(rng.integers(100, 400)) if faulty else T
    kk, cc = np.full(T, k), np.full(T, c)
    if faulty:
        if rng.random() < 0.5:
            kk[tf:] = k * rng.choice([0.5, 1.6])
        else:
            cc[tf:] = c * rng.choice([0.2, 3.0])
    u = rng.normal(0, 1, T)                       # random excitation
    meas, proc = rng.normal(0, 0.05, T), rng.normal(0, 0.01, T)
    x = v = 0.0
    y = np.zeros(T)
    for i in range(T):
        y[i] = x + meas[i]                        # noisy output
        x, v = x + DT * v, v + DT * (u[i] - cc[i] * v - kk[i] * x) + proc[i]
    return y, u, tf


# ---- data: simulate, then split episodes ONCE (stratified by fault status) ----
faulty = rng.random(N_EP) < 0.5
sims = [simulate(rng, f) for f in faulty]
Y = np.array([s[0] for s in sims]); U = np.array([s[1] for s in sims])
TF = np.array([s[2] for s in sims])
idx_tr, idx_te = train_test_split(np.arange(N_EP), test_size=0.25, stratify=faulty, random_state=SEED)
print(f"[data] {N_EP} episodes x {T} samples; train eps={len(idx_tr)}, test eps={len(idx_te)}")


# ---- baseline: nominal ARX(2,2) fit on fault-free TRAINING episodes ----
def arx_regressors(Y, U):
    """Regressors [y(t-1), y(t-2), u(t-1), u(t-2), 1] and targets y(t) for t = 2..T-1."""
    X = np.stack([Y[:, 1:-1], Y[:, :-2], U[:, 1:-1], U[:, :-2], np.ones_like(Y[:, 1:-1])], -1)
    return X, Y[:, 2:]


def residuals(theta, Y, U):
    X, y = arx_regressors(Y, U)
    r = np.zeros_like(Y)
    r[:, 2:] = y - X @ theta
    return r


def causal_mean(a, w):
    """Causal moving average over the last w samples (no look-ahead)."""
    c = np.cumsum(np.pad(a, ((0, 0), (w, 0))), axis=1)
    return (c[:, w:] - c[:, :-w]) / w


ff_tr = idx_tr[~faulty[idx_tr]]
Xa, ya = arx_regressors(Y[ff_tr], U[ff_tr])
theta = np.linalg.lstsq(Xa.reshape(-1, 5), ya.reshape(-1), rcond=None)[0]
R = residuals(theta, Y, U)
S = causal_mean(np.abs(R), SMOOTH)
thr = np.quantile(S[ff_tr, 2:], 0.995)            # calibrated on fault-free training residuals only
print(f"[baseline] ARX fitted on {len(ff_tr)} fault-free train episodes; threshold={thr:.4f}")


# ---- improved: sliding-window features and class-balanced LightGBM ----
def window_feats(R, Y, U):
    """Features over each length-W window ending at sample t >= W-1: shape (N, T-W+1, n_feat)."""
    def win(a):
        return sliding_window_view(a, W, axis=1)
    r, ar, y, u = win(R), win(np.abs(R)), win(Y), win(U)
    F = [r.mean(-1), r.std(-1), r.min(-1), r.max(-1), r[..., -1], ar.mean(-1), ar.max(-1),
         y.mean(-1), y.std(-1), y[..., -1] - y[..., 0], u.mean(-1), u.std(-1)]
    return np.stack(F, -1)


tidx = np.arange(W - 1, T)                        # window end-times; shared by both detectors
F_all = window_feats(R, Y, U)
nf = F_all.shape[-1]
LAB = (tidx[None, :] >= TF[:, None]).astype(int)  # window is faulty once its end is after onset
Ftr = F_all[idx_tr].reshape(-1, nf)
ytr = LAB[idx_tr].ravel()
gtr = np.repeat(idx_tr, len(tidx))                # episode id per row, for grouped CV

# small grid, selected ONLY by grouped CV on the training episodes
best, best_f1 = None, -1.0
for p in ParameterGrid({"num_leaves": [15, 31], "n_estimators": [200, 400]}):
    f1s = []
    for a, b in GroupKFold(n_splits=3).split(Ftr, ytr, gtr):
        m = lgb.LGBMClassifier(class_weight="balanced", learning_rate=0.05, random_state=SEED,
                               verbose=-1, **p).fit(Ftr[a], ytr[a])
        f1s.append(f1_score(ytr[b], m.predict(Ftr[b]), pos_label=1, zero_division=0))
    cv_f1 = float(np.mean(f1s))
    print(f"[cv] {p} -> CV fault-F1={cv_f1:.4f}")
    if cv_f1 > best_f1:
        best, best_f1 = p, cv_f1
print(f"[cv] selected {best}")


# ---- shared evaluation (test set, used once per model) ----
def evaluate(flag, idx):
    """flag: bool alarms (n_episodes, n_windows) for episodes idx. Returns F1 (fault class),
    false alarm rate on fault-free windows, missed-fault rate and mean detection delay."""
    lab = tidx[None, :] >= TF[idx][:, None]
    fa = faulty[idx]
    tf_f = TF[idx][fa]
    post = flag[fa] & (tidx[None, :] >= tf_f[:, None])
    det = post.any(1)
    delay = np.where(det, tidx[post.argmax(1)] - tf_f, T - tf_f)  # missed: remaining episode length
    return {"f1": f1_score(lab.ravel(), flag.ravel(), pos_label=1, zero_division=0),
            "far": flag[~fa].mean(), "miss_rate": 1 - det.mean(), "delay": delay.mean()}


base = evaluate(S[idx_te][:, W - 1:] > thr, idx_te)   # deterministic: same result for every seed
base_runs, imp_runs = [], []
for seed in SEEDS:
    m = lgb.LGBMClassifier(class_weight="balanced", learning_rate=0.05, random_state=seed,
                           verbose=-1, **best).fit(Ftr, ytr)
    pr = m.predict_proba(F_all[idx_te].reshape(-1, nf))[:, 1].reshape(len(idx_te), -1) >= 0.5
    imp_runs.append(evaluate(pr, idx_te))
    base_runs.append(base)
    print(f"[test] seed {seed}: baseline F1={base['f1']:.4f}  improved F1={imp_runs[-1]['f1']:.4f}")

agg = lambda runs, k: float(np.mean([r[k] for r in runs]))
sd = lambda runs, k: float(np.std([r[k] for r in runs]))
b_f1, i_f1 = agg(base_runs, "f1"), agg(imp_runs, "f1")
metrics = {
    "baseline_score": b_f1, "improved_score": i_f1, "improvement": i_f1 - b_f1,
    "baseline_score_std": sd(base_runs, "f1"), "improved_score_std": sd(imp_runs, "f1"),
    "baseline_far": agg(base_runs, "far"), "improved_far": agg(imp_runs, "far"),
    "baseline_miss_rate": agg(base_runs, "miss_rate"), "improved_miss_rate": agg(imp_runs, "miss_rate"),
    "baseline_delay_samples": agg(base_runs, "delay"), "improved_delay_samples": agg(imp_runs, "delay"),
    "baseline_threshold": float(thr), "cv_best_fault_f1": best_f1,
}
if not all(np.isfinite(v) for v in metrics.values()):
    print("Non-finite metrics; failing."); sys.exit(1)

# ---- one comparison chart ----
os.makedirs("figures", exist_ok=True)
names = ["F1 (fault)", "False alarm rate", "Missed-fault rate"]
bv = [b_f1, metrics["baseline_far"], metrics["baseline_miss_rate"]]
iv = [i_f1, metrics["improved_far"], metrics["improved_miss_rate"]]
xs = np.arange(3)
plt.figure(figsize=(7, 4))
plt.bar(xs - 0.2, bv, 0.4, label="baseline (ARX residual threshold)")
plt.bar(xs + 0.2, iv, 0.4, label="improved (balanced LightGBM)")
plt.xticks(xs, names); plt.ylim(0, 1.05); plt.legend(fontsize=8); plt.title("Test-set comparison")
plt.tight_layout(); plt.savefig("figures/comparison.png", dpi=120); plt.close()

# ---- results.json ----
n_w = len(tidx)
results = {
    "question": "Can a class-balanced LightGBM classifier on residual windows detect abrupt parameter faults "
                "in a simulated dynamic system with fewer missed faults and false alarms than a standard "
                "model-based residual-threshold detector?",
    "method": "Seeded numpy simulation of a discrete-time mass-spring-damper with abrupt stiffness/damping "
              "faults; episode-level stratified train/test split; grouped-CV tuning on train only; "
              "window-level evaluation on unseen test episodes.",
    "dataset": {"name": "Simulated mass-spring-damper (numpy, seed 42)",
                "source_url": "synthetic: generated in analysis.py, no download",
                "n_rows": int(N_EP * n_w), "n_train": int(len(idx_tr) * n_w), "n_test": int(len(idx_te) * n_w),
                "same_as_paper": False, "baseline_is_papers_method": False},
    "metric_name": "F1 for fault class on test windows",
    "higher_is_better": True,
    "baseline": "Nominal ARX(2,2) one-step-ahead predictor fit by least squares on fault-free training "
                "episodes; |residual| smoothed causally over 10 samples; alarm when above the 99.5th percentile "
                "of fault-free training residuals.",
    "improved": "Class-balanced LightGBM classifier on sliding-window (length 20) features of residual, |residual|, "
                "output and input statistics; labels faulty once the window ends after fault onset; "
                "hyper-parameters chosen by grouped 3-fold CV on training episodes.",
    "metrics": metrics,
    "notes": f"Single simulated system and fault family; {N_EP} episodes of {T} samples, 25% held out by episode. "
             "Baseline has no randomness, so its std is 0. Improved std is across LightGBM seeds 42-44 only. "
             "Detection delay counts missed faults as the remaining episode length. Windows before sample 19 are "
             "not scored for either model. Results are specific to this simulator and are not the paper's result.",
}
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"[done] results.json written in {time.time() - t_start:.1f}s")
