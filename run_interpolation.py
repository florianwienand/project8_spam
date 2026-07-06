"""Kernel interpolation K_alpha = alpha*K_spectrum + (1-alpha)*K_edit, swept
from pure edit (alpha=0) to pure spectrum (alpha=1), evaluated clean and under
attack. Every alpha gets the identical kPCA + linear-SVM treatment (the mixture
is indefinite for alpha < 1), so the endpoints reproduce the main-table rows.

Run from the project root:  python run_interpolation.py
"""
import json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.preprocessing import load_sms, clean_text, make_split, obfuscate
from src.kernels import SpectrumKernel, LevenshteinKernel
from courselib.models.svm import BinaryKernelSVM

SEEDS = [0, 1, 2, 3, 4]
DATA_PATH = "data/SMSSpamCollection"
SUBSAMPLE, TRAIN_FRAC = 1000, 0.75
K, SIGMA, C = 3, 0.45, 1.0
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
VAR_KEEP = 0.99


def kpca_basis(K_tr, var_keep=VAR_KEEP):
    """Positive eigen-directions covering `var_keep` of the positive mass.
    Same recipe as levenshtein_kpca_features, factored out for the mixture."""
    Ks = (K_tr + K_tr.T) / 2.0
    w, V = np.linalg.eigh(Ks)
    pos = w > 1e-8
    w, V = w[pos], V[:, pos]
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    cum = np.cumsum(w) / w.sum()
    keep = max(1, min(int(np.searchsorted(cum, var_keep)) + 1, len(w)))
    return V[:, :keep], w[:keep]


def rec_prec(pred, y):
    tp = np.sum((pred == 1) & (y == 1)); fn = np.sum((pred == -1) & (y == 1))
    fp = np.sum((pred == 1) & (y == -1))
    return (tp / (tp + fn) if (tp + fn) else 0.0,
            tp / (tp + fp) if (tp + fp) else 0.0)


def main():
    t0 = time.time()
    X_raw, y = load_sms(DATA_PATH)
    X = np.array([clean_text(t) for t in X_raw], dtype=object)
    print(f"interpolation | alphas {ALPHAS} | {len(SEEDS)} seeds | levels {LEVELS}\n")

    nA, nL = len(ALPHAS), len(LEVELS)
    clean_acc = np.zeros((len(SEEDS), nA)); clean_rec = np.zeros((len(SEEDS), nA))
    clean_prec = np.zeros((len(SEEDS), nA))
    att_rec = np.zeros((len(SEEDS), nA, nL))

    spec = SpectrumKernel(k=K)
    lev = LevenshteinKernel(sigma=SIGMA, psd_fix=None)

    for si, seed in enumerate(SEEDS):
        X_tr, y_tr, X_te, y_te = make_split(X, y, SUBSAMPLE, TRAIN_FRAC, seed=seed)
        spam_idx = np.where(y_te == 1)[0]
        ham_idx = np.where(y_te == -1)[0]

        # train Grams and clean test rectangles once; every alpha is a weighted sum
        Ksp_tr = spec.normalized(X_tr, X_tr)
        Klv_tr = lev(X_tr, X_tr)
        Ksp_te = spec.normalized(X_te, X_tr)
        Klv_te = lev(X_te, X_tr)

        # attacked rectangles once per level (they do not depend on alpha);
        # same attack stream as run_obfuscation
        rng = np.random.default_rng(10_000 + seed)
        Ksp_att, Klv_att = [], []
        for level in LEVELS:
            obf = [obfuscate(X_te[i], level, rng) for i in spam_idx]
            obf = np.array(obf, dtype=object)
            Ksp_att.append(spec.normalized(obf, X_tr))
            Klv_att.append(lev(obf, X_tr))

        for ai, a in enumerate(ALPHAS):
            V, w = kpca_basis(a * Ksp_tr + (1 - a) * Klv_tr)
            Phi_tr = V * np.sqrt(w)
            proj = lambda Kx: Kx @ V / np.sqrt(w)
            svm = BinaryKernelSVM(C=C, kernel="linear")
            svm.fit(Phi_tr, y_tr)

            pred = svm(proj(a * Ksp_te + (1 - a) * Klv_te))
            r, p = rec_prec(pred, y_te)
            clean_acc[si, ai] = np.mean(pred == y_te) * 100
            clean_rec[si, ai], clean_prec[si, ai] = r, p

            fp_clean = int(np.sum(pred[ham_idx] == 1))   # ham is never attacked
            for li in range(nL):
                pr = svm(proj(a * Ksp_att[li] + (1 - a) * Klv_att[li]))
                att_rec[si, ai, li] = np.mean(pr == 1)
        print(f"  seed {seed} done ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'alpha':>6}{'clean acc %':>14}{'clean recall':>14}{'recall@0.3':>12}{'recall@0.6':>12}")
    for ai, a in enumerate(ALPHAS):
        print(f"{a:>6.2f}{clean_acc[:, ai].mean():>11.1f}±{clean_acc[:, ai].std():<4.1f}"
              f"{clean_rec[:, ai].mean():>10.2f}±{clean_rec[:, ai].std():<4.2f}"
              f"{att_rec[:, ai, 3].mean():>9.2f}±{att_rec[:, ai, 3].std():<4.2f}"
              f"{att_rec[:, ai, 6].mean():>9.2f}±{att_rec[:, ai, 6].std():<4.2f}")

    out = {"alphas": ALPHAS, "levels": LEVELS, "seeds": SEEDS,
           "clean_acc": {"mean": clean_acc.mean(0).tolist(), "std": clean_acc.std(0).tolist()},
           "clean_recall": {"mean": clean_rec.mean(0).tolist(), "std": clean_rec.std(0).tolist()},
           "clean_precision": {"mean": clean_prec.mean(0).tolist()},
           "attacked_recall_mean": att_rec.mean(0).tolist(),   # (alpha, level)
           "attacked_recall_std": att_rec.std(0).tolist()}
    with open("results/interpolation.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved results/interpolation.json")

    # (A) recall vs level, one line per alpha; (B) the tradeoff vs alpha
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    cmap = plt.cm.viridis(np.linspace(0.08, 0.92, nA))
    for ai, a in enumerate(ALPHAS):
        m, s = att_rec[:, ai, :].mean(0), att_rec[:, ai, :].std(0)
        axA.plot(LEVELS, m, "-o", color=cmap[ai], lw=1.8, ms=4,
                 label=f"α={a:.2f}" + ("  (pure edit)" if a == 0 else "  (pure spectrum)" if a == 1 else ""))
        axA.fill_between(LEVELS, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), color=cmap[ai], alpha=0.12)
    axA.set(xlabel="obfuscation level", ylabel="spam recall", ylim=(0, 1.02))
    axA.set_title("attacked recall, one line per mixture", fontsize=11)
    axA.grid(alpha=0.3); axA.legend(fontsize=8)

    axB.errorbar(ALPHAS, clean_acc.mean(0) / 100, yerr=clean_acc.std(0) / 100, fmt="-s",
                 color="#333333", lw=1.8, ms=5, capsize=3, label="clean accuracy")
    axB.errorbar(ALPHAS, clean_rec.mean(0), yerr=clean_rec.std(0), fmt="-o",
                 color="#1f77b4", lw=1.8, ms=5, capsize=3, label="clean recall")
    axB.errorbar(ALPHAS, att_rec[:, :, 6].mean(0), yerr=att_rec[:, :, 6].std(0), fmt="-^",
                 color="#d62728", lw=1.8, ms=5, capsize=3, label="recall @ attack 0.6")
    axB.set(xlabel="α   (0 = pure edit,  1 = pure spectrum)", ylabel="score", ylim=(0, 1.05))
    axB.set_title("the tradeoff: clean vs attacked, per mixture", fontsize=11)
    axB.grid(alpha=0.3); axB.legend(fontsize=9)
    fig.suptitle("Kernel interpolation  αK_spectrum + (1−α)K_edit   (mean ± 1 std over 5 seeds)",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("figures/kernel_interpolation.png", dpi=130, bbox_inches="tight")
    print("saved figures/kernel_interpolation.png")

    # standalone copy of panel (B) sized for the report's one-column slot
    figB, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.errorbar(ALPHAS, clean_acc.mean(0) / 100, yerr=clean_acc.std(0) / 100, fmt="-s",
                color="#333333", lw=1.8, ms=5, capsize=3, label="clean accuracy")
    ax.errorbar(ALPHAS, clean_rec.mean(0), yerr=clean_rec.std(0), fmt="-o",
                color="#1f77b4", lw=1.8, ms=5, capsize=3, label="clean recall")
    ax.errorbar(ALPHAS, att_rec[:, :, 6].mean(0), yerr=att_rec[:, :, 6].std(0), fmt="-^",
                color="#d62728", lw=1.8, ms=5, capsize=3, label="recall @ attack 0.6")
    ax.set(xlabel="α   (0 = pure edit,  1 = pure spectrum)", ylabel="score", ylim=(0, 1.05))
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("figures/kernel_interpolation_report.png", dpi=130, bbox_inches="tight")
    print("saved figures/kernel_interpolation_report.png")


if __name__ == "__main__":
    main()