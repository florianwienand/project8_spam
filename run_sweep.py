"""Kernel hyperparameter sensitivity -- the supplementary figure. Its job is to turn three
claims the report makes in prose into evidence you can actually see, all on the same
1000-message subsample / 5 seeds as the main experiment so the numbers line up:

  1. Levenshtein sigma, geometry: as sigma -> 0 the kernel collapses toward the identity
     (mean off-diagonal -> 0, support-vector fraction -> 1); as sigma grows the raw Gram
     gets more indefinite (negative-eigenvalue mass rises).
  2. Levenshtein sigma, performance: the kPCA classifier's *ranking* (ROC-AUC) is flat
     across sigma, while its *recall* at the default threshold peaks in a mid-sigma band
     -- the same ranking-vs-threshold split as the main results. sigma=0.45 sits in the band.
  3. Spectrum order k: larger k is more specific (spam/between block ratio rises) and
     sparser (mean off-diagonal falls, #SV rises), but recall is an inverted-U peaking at
     k=3; precision stays ~1.0 throughout.

The one trick that keeps this cheap: the normalized Levenshtein distance matrix is the
expensive bit and it doesn't depend on sigma, so we compute it ONCE per seed and only
re-apply the (cheap) RBF transform per sigma. Raw per-seed arrays go to results/sweep.json,
the figure to figures/kernel_sensitivity.png.

Run from the project root:  python run_sweep.py
"""
import json, time
import numpy as np
import matplotlib.pyplot as plt
from rapidfuzz.process import cdist
from rapidfuzz.distance import Levenshtein
from sklearn.metrics import roc_auc_score

from src.preprocessing import load_sms, clean_text, make_split
from src.kernels import SpectrumKernel
from src.evaluation import class_block_means
from courselib.models.svm import BinaryKernelSVM

SEEDS = [0, 1, 2, 3, 4]
SUBSAMPLE, TRAIN_FRAC, C = 1000, 0.75, 1.0
SIGMAS = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
KS = [1, 2, 3, 4, 5, 6]
SIGMA_REF, K_REF = 0.45, 3


def _kpca_from_D(D_tr, D_te, sigma, var_keep=0.99):
    """levenshtein_kpca_features, but reusing a precomputed distance matrix so the sigma
    loop doesn't recompute it every time. same projection, just cheaper to sweep."""
    K = np.exp(-(D_tr ** 2) / (2 * sigma ** 2))
    K = (K + K.T) / 2.0
    w, V = np.linalg.eigh(K)
    neg = w[w < 0]
    neg_mass = float(np.sum(np.abs(neg)) / np.sum(np.abs(w)))
    mean_off = float(K[~np.eye(len(K), dtype=bool)].mean())
    pos = w > 1e-8
    w, V = w[pos], V[:, pos]
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    cum = np.cumsum(w) / w.sum()
    keep = max(1, min(int(np.searchsorted(cum, var_keep)) + 1, len(w)))
    w, V = w[:keep], V[:, :keep]
    Phi_tr = V * np.sqrt(w)
    Phi_te = np.exp(-(D_te ** 2) / (2 * sigma ** 2)) @ V / np.sqrt(w)
    return Phi_tr, Phi_te, mean_off, neg_mass


def _rec_prec(pred, y):
    tp = np.sum((pred == 1) & (y == 1)); fn = np.sum((pred == -1) & (y == 1))
    fp = np.sum((pred == 1) & (y == -1))
    return (tp / (tp + fn) if (tp + fn) else 0.0,
            tp / (tp + fp) if (tp + fp) else 0.0)


def compute():
    t0 = time.time()
    X_raw, y = load_sms("data/SMSSpamCollection")
    X = np.array([clean_text(t) for t in X_raw], dtype=object)

    lev = {s: {k: [] for k in ("recall", "precision", "auc", "sv_frac",
                               "mean_off", "neg_mass")} for s in SIGMAS}
    spec = {kk: {k: [] for k in ("recall", "precision", "auc", "sv",
                                 "mean_off", "spam_over_between")} for kk in KS}

    for seed in SEEDS:
        X_tr, y_tr, X_te, y_te = make_split(X, y, SUBSAMPLE, TRAIN_FRAC, seed=seed)
        n_tr = len(X_tr)
        D_tr = cdist(list(X_tr), list(X_tr), scorer=Levenshtein.normalized_distance, workers=-1)
        D_te = cdist(list(X_te), list(X_tr), scorer=Levenshtein.normalized_distance, workers=-1)

        for s in SIGMAS:
            Phi_tr, Phi_te, mean_off, neg_mass = _kpca_from_D(D_tr, D_te, s)
            svm = BinaryKernelSVM(C=C, kernel="linear"); svm.fit(Phi_tr, y_tr)
            r, p = _rec_prec(svm(Phi_te), y_te)
            lev[s]["recall"].append(r); lev[s]["precision"].append(p)
            lev[s]["auc"].append(float(roc_auc_score((y_te == 1).astype(int),
                                                     svm.decision_function(Phi_te))))
            lev[s]["sv_frac"].append(len(svm.alphas) / n_tr)
            lev[s]["mean_off"].append(mean_off); lev[s]["neg_mass"].append(neg_mass)

        for kk in KS:
            svm = BinaryKernelSVM(C=C, kernel="custom",
                                  kernel_function=SpectrumKernel(k=kk).normalized)
            svm.fit(X_tr, y_tr)
            r, p = _rec_prec(svm(X_te), y_te)
            order = np.argsort(y_tr)
            Kk = SpectrumKernel(k=kk).normalized(X_tr[order], X_tr[order])
            blk = class_block_means(Kk, y_tr[order])
            spec[kk]["recall"].append(r); spec[kk]["precision"].append(p)
            spec[kk]["auc"].append(float(roc_auc_score((y_te == 1).astype(int),
                                                       svm.decision_function(X_te))))
            spec[kk]["sv"].append(int(len(svm.alphas)))
            spec[kk]["mean_off"].append(float(Kk[~np.eye(len(Kk), dtype=bool)].mean()))
            spec[kk]["spam_over_between"].append(float(blk["spam_over_between"]))
        print(f"  seed {seed} done ({time.time()-t0:.0f}s)", flush=True)

    out = {"seeds": SEEDS, "sigmas": SIGMAS, "ks": KS,
           "sigma_ref": SIGMA_REF, "k_ref": K_REF,
           "lev_sigma": {str(s): lev[s] for s in SIGMAS},
           "spectrum_k": {str(kk): spec[kk] for kk in KS}}
    with open("results/sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved results/sweep.json", flush=True)
    return out


def _ms(section, xs, key):
    m = np.array([np.mean(section[str(x)][key]) for x in xs])
    s = np.array([np.std(section[str(x)][key]) for x in xs])
    return m, s


def plot(d):
    S = np.array(d["sigmas"]); Ks = np.array(d["ks"])
    sref, kref = d["sigma_ref"], d["k_ref"]
    lev, spec = d["lev_sigma"], d["spectrum_k"]
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15, 4.6))

    # A: sigma geometry
    off_m, _ = _ms(lev, S, "mean_off"); sv_m, _ = _ms(lev, S, "sv_frac")
    neg_m, _ = _ms(lev, S, "neg_mass")
    axA.plot(S, off_m, "-o", color="#1f77b4", lw=1.8, ms=4, label="mean off-diagonal")
    axA.plot(S, sv_m, "-s", color="#9467bd", lw=1.8, ms=4, label="support-vector fraction")
    axA.set_ylim(0, 1.02); axA.set_xlabel("Levenshtein width  σ"); axA.set_ylabel("value  (0–1)")
    axA.axvline(sref, color="grey", ls="--", lw=1)
    axA.text(sref + 0.01, 0.94, f"σ={sref}", color="grey", fontsize=8)
    axAr = axA.twinx()
    axAr.plot(S, neg_m, "-^", color="#d62728", lw=1.8, ms=4, label="neg-eigenvalue mass")
    axAr.set_ylabel("neg-eigenvalue mass", color="#d62728")
    axAr.tick_params(axis="y", labelcolor="#d62728"); axAr.set_ylim(0, max(neg_m) * 1.4)
    l1, la = axA.get_legend_handles_labels(); l2, lb = axAr.get_legend_handles_labels()
    axA.legend(l1 + l2, la + lb, fontsize=7.5, loc="center right")
    axA.set_title("σ geometry: small σ → identity (off-diag→0, all SVs);\nlarge σ → more indefinite", fontsize=9.5)
    axA.grid(alpha=0.3)

    # B: sigma performance
    rec_m, rec_s = _ms(lev, S, "recall"); auc_m, auc_s = _ms(lev, S, "auc")
    axB.axvspan(0.35, 0.65, color="green", alpha=0.06)
    axB.plot(S, rec_m, "-o", color="#2ca02c", lw=1.8, ms=4, label="kPCA recall")
    axB.fill_between(S, rec_m - rec_s, rec_m + rec_s, color="#2ca02c", alpha=0.15)
    axB.plot(S, auc_m, "-D", color="#1f77b4", lw=1.8, ms=4, label="kPCA ROC-AUC")
    axB.fill_between(S, auc_m - auc_s, auc_m + auc_s, color="#1f77b4", alpha=0.15)
    axB.axvline(sref, color="grey", ls="--", lw=1)
    axB.text(sref + 0.01, 0.05, f"σ={sref}", color="grey", fontsize=8)
    axB.set_ylim(0, 1.02); axB.set_xlabel("Levenshtein width  σ"); axB.set_ylabel("score")
    axB.legend(fontsize=8, loc="center right")
    axB.set_title("σ performance: ranking (AUC) flat across σ,\nrecall peaks in a mid-σ band", fontsize=9.5)
    axB.grid(alpha=0.3)

    # C: spectrum k tradeoff
    rk_m, rk_s = _ms(spec, Ks, "recall"); pk_m, pk_s = _ms(spec, Ks, "precision")
    sb_m, _ = _ms(spec, Ks, "spam_over_between")
    axC.plot(Ks, rk_m, "-o", color="#2ca02c", lw=1.8, ms=5, label="recall")
    axC.fill_between(Ks, rk_m - rk_s, rk_m + rk_s, color="#2ca02c", alpha=0.15)
    axC.plot(Ks, pk_m, "-s", color="#ff7f0e", lw=1.8, ms=5, label="precision")
    axC.fill_between(Ks, pk_m - pk_s, pk_m + pk_s, color="#ff7f0e", alpha=0.15)
    axC.set_ylim(0, 1.05); axC.set_xlabel("spectrum order  k"); axC.set_ylabel("recall / precision")
    axC.axvline(kref, color="grey", ls="--", lw=1)
    axC.text(kref + 0.05, 0.05, f"k={kref}", color="grey", fontsize=8)
    axCr = axC.twinx()
    axCr.plot(Ks, sb_m, "-^", color="#8c564b", lw=1.8, ms=5, label="spam/between ratio")
    axCr.set_ylabel("spam/between block ratio", color="#8c564b")
    axCr.tick_params(axis="y", labelcolor="#8c564b"); axCr.set_ylim(1, max(sb_m) * 1.1)
    l1, la = axC.get_legend_handles_labels(); l2, lb = axCr.get_legend_handles_labels()
    axC.legend(l1 + l2, la + lb, fontsize=8, loc="center left")
    axC.set_title("k tradeoff: higher k → more specific (↑ ratio)\nbut recall peaks at k=3", fontsize=9.5)
    axC.grid(alpha=0.3)

    fig.suptitle("Kernel hyperparameter sensitivity (mean over 5 seeds, band = ±1 std)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("figures/kernel_sensitivity.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("saved figures/kernel_sensitivity.png", flush=True)


def main():
    plot(compute())


if __name__ == "__main__":
    main()