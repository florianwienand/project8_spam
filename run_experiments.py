"""Main comparison: spectrum and Levenshtein string kernels vs a word TF-IDF +
linear-SVM baseline, plus an MLP on the same TF-IDF features as the second
course model. All numbers are mean +/- std over 5 stratified splits.

Run from the project root:  python run_experiments.py"""
import json
import types
from collections import defaultdict
import numpy as np

from src.preprocessing import load_sms, clean_text, make_split, tfidf_features
from src.kernels import SpectrumKernel, LevenshteinKernel, levenshtein_kpca_features
from src.evaluation import binary_metrics, plot_kernel_comparison, class_block_means, plot_roc_pr
from courselib.models.svm import BinaryKernelSVM
from courselib.models.nn import MLP
from courselib.optimizers import GDOptimizer
from courselib.utils.preprocessing import labels_encoding

SEEDS = [0, 1, 2, 3, 4]
FIGURE_SEED = 0           # split used for the heatmap figure and PSD diagnostic
DATA_PATH = "data/SMSSpamCollection"
SUBSAMPLE = 1000
TRAIN_FRAC = 0.75
K = 3                     # spectrum kernel order
SIGMA = 0.45              # Levenshtein RBF width
C = 1.0

MLP_HIDDEN = 64
MLP_LR = 1.0
MLP_EPOCHS = 300
MLP_BATCH = 64


def _fast_dense_loss_grad(self, X_prev, delta):
    """Same gradient as courselib's DenseLayer.loss_grad, but as one matmul
    instead of averaging an (N, out, in) tensor. Numerically identical, ~10x faster."""
    return {f'{self.name}_W': (X_prev.T @ delta) / X_prev.shape[0],
            f'{self.name}_b': np.mean(delta, axis=0, keepdims=True)}


class MLPClassifier:
    """courselib MLP wrapped as a +1/-1 classifier with the same interface as
    BinaryKernelSVM. decision_function returns P(spam) - P(ham), so its zero
    threshold matches the argmax decision and the SVM's sign convention."""

    def __init__(self, hidden=MLP_HIDDEN, lr=MLP_LR, epochs=MLP_EPOCHS,
                 batch_size=MLP_BATCH, seed=0):
        self.hidden, self.lr = hidden, lr
        self.epochs, self.batch_size, self.seed = epochs, batch_size, seed
        self.net, self.alphas = None, None

    def fit(self, X, y):
        np.random.seed(self.seed)                # init + batch shuffle use global RNG
        opt = GDOptimizer(learning_rate=self.lr)
        self.net = MLP(widths=[X.shape[1], self.hidden, 2], optimizer=opt,
                       activation="ReLU", output_activation="Softmax", loss="CE")
        for layer in self.net.layers:
            layer.loss_grad = types.MethodType(_fast_dense_loss_grad, layer)
        Y = labels_encoding(y, labels=[-1, 1], pos_value=1, neg_value=0)
        self.net.fit(X, Y, num_epochs=self.epochs, batch_size=self.batch_size)
        return self

    def decision_function(self, X):
        p = self.net.decision_function(X)
        return p[:, 1] - p[:, 0]

    def __call__(self, X):
        return np.where(self.decision_function(X) > 0, 1, -1)


def evaluate(X, y, seed):
    """Run every model on one stratified split."""
    X_tr, y_tr, X_te, y_te = make_split(X, y, SUBSAMPLE, TRAIN_FRAC, seed=seed)
    majority = max((y_te == 1).mean(), (y_te == -1).mean()) * 100
    metrics, sv, scores = {}, {}, {}

    def record(name, model, pred_input):
        metrics[name] = binary_metrics(model(pred_input), y_te)
        sv[name] = len(model.alphas) if getattr(model, "alphas", None) is not None else None
        scores[name] = (np.asarray(model.decision_function(pred_input)), y_te)

    # baseline: word TF-IDF + linear SVM
    Xtr_t, Xte_t, _ = tfidf_features(X_tr, X_te)
    base = BinaryKernelSVM(C=C, kernel="linear")
    base.fit(Xtr_t, y_tr)
    record("TF-IDF + linear (baseline)", base, Xte_t)

    # second course model on the same features
    mlp = MLPClassifier(seed=seed)
    mlp.fit(Xtr_t, y_tr)
    record("MLP (TF-IDF)", mlp, Xte_t)

    # string kernels through the dual solver; spectrum kernels are cosine-normalized
    # (removes the length bias, keeps a single C comparable across models)
    direct = {
        "Spectrum k=3":                          SpectrumKernel(k=3).normalized,
        "Spectrum k=2":                          SpectrumKernel(k=2).normalized,
        f"Levenshtein direct(shift) s={SIGMA}":  LevenshteinKernel(sigma=SIGMA),
    }
    for name, kfn in direct.items():
        svm = BinaryKernelSVM(C=C, kernel="custom", kernel_function=kfn)
        svm.fit(X_tr, y_tr)
        record(name, svm, X_te)

    # Levenshtein via kPCA projection + linear SVM
    Phi_tr, Phi_te = levenshtein_kpca_features(X_tr, X_te, sigma=SIGMA)
    lev = BinaryKernelSVM(C=C, kernel="linear")
    lev.fit(Phi_tr, y_tr)
    record(f"Levenshtein kPCA s={SIGMA}", lev, Phi_te)

    # sorted training Grams for the block table and heatmap
    order = np.argsort(y_tr)
    Xo, yo = X_tr[order], y_tr[order]
    n_ham = int((yo == -1).sum())
    Ksp = SpectrumKernel(k=K).normalized(Xo, Xo)
    Klv = LevenshteinKernel(sigma=SIGMA, psd_fix=None)(Xo, Xo)
    blocks = {"spectrum": class_block_means(Ksp, yo),
              "levenshtein": class_block_means(Klv, yo)}
    return metrics, sv, blocks, (Ksp, Klv, n_ham), majority, scores


def mean_std(vals):
    return float(np.mean(vals)), float(np.std(vals))


def main():
    X_raw, y = load_sms(DATA_PATH)
    X = np.array([clean_text(t) for t in X_raw], dtype=object)
    print(f"loaded {len(X)} messages | spam={np.sum(y == 1)} ham={np.sum(y == -1)}")
    print(f"averaging over {len(SEEDS)} seeds: {SEEDS}\n")

    # PSD diagnostic on the representative split
    X_tr0, _, _, _ = make_split(X, y, SUBSAMPLE, TRAIN_FRAC, seed=FIGURE_SEED)
    d = LevenshteinKernel(sigma=SIGMA).diagnostics(X_tr0)
    print(f"Levenshtein PSD report (seed={FIGURE_SEED}, sigma={SIGMA}): "
          f"lambda_min={d['lambda_min']:.3f}, {d['n_negative']}/{d['n']} negative eigenvalues, "
          f"neg-mass={d['neg_eig_mass']:.3f}, shift rho={d['shift_rho']:.3f} "
          f"(diagonal perturbation {d['shift_rel_perturbation']:.3f} -- over-regularizes)\n")

    m_acc = defaultdict(lambda: defaultdict(list))
    m_sv = defaultdict(list)
    b_acc = defaultdict(lambda: defaultdict(list))
    roc_data = defaultdict(list)
    majorities, fig_grams = [], None
    for seed in SEEDS:
        metrics, sv, blocks, grams, majority, scores = evaluate(X, y, seed)
        for name, mm in metrics.items():
            for key in ("accuracy", "precision", "recall"):
                m_acc[name][key].append(mm[key])
            m_sv[name].append(sv[name])
        for kern, bl in blocks.items():
            for stat, val in bl.items():
                b_acc[kern][stat].append(val)
        for name, sc in scores.items():
            roc_data[name].append(sc)
        majorities.append(majority)
        if seed == FIGURE_SEED:
            fig_grams = grams
        print(f"  seed {seed} done")

    maj_m, maj_s = mean_std(majorities)

    # results table at the default threshold
    print(f"\n{'model':<32}{'accuracy %':>15}{'precision':>15}{'recall':>15}{'#SV':>13}")
    agg_metrics = {}
    for name in m_acc:
        a = mean_std(m_acc[name]["accuracy"]); p = mean_std(m_acc[name]["precision"])
        r = mean_std(m_acc[name]["recall"])
        sv_vals = [v for v in m_sv[name] if v is not None]
        s = mean_std(sv_vals) if sv_vals else None
        agg_metrics[name] = {"accuracy": a, "precision": p, "recall": r, "n_sv": s}
        sv_str = f"{s[0]:>7.0f}±{s[1]:<4.0f}" if s is not None else f"{'n/a':>7}    "
        print(f"{name:<32}{a[0]:>8.1f}±{a[1]:<5.1f}{p[0]:>8.2f}±{p[1]:<5.2f}"
              f"{r[0]:>8.2f}±{r[1]:<5.2f}{sv_str}")
    print(f"\n(majority-class accuracy = {maj_m:.1f}±{maj_s:.1f}% -- a model at or below this is degenerate)")

    # threshold-independent ranking: ROC + PR curves
    auc_summary = plot_roc_pr(roc_data, "figures/roc_pr_curves.png",
                              prevalence=float((y == 1).mean()))
    print(f"\n{'model':<32}{'ROC-AUC':>16}{'avg-precision':>18}")
    for name, s in auc_summary.items():
        print(f"{name:<32}{s['roc_auc'][0]:>9.3f}±{s['roc_auc'][1]:<5.3f}"
              f"{s['avg_precision'][0]:>11.3f}±{s['avg_precision'][1]:<5.3f}")

    # class-block similarity table
    print(f"\n{'kernel':<22}{'within-spam':>16}{'within-ham':>16}{'between':>16}{'spam/between':>16}")
    agg_blocks = {}
    for kern in b_acc:
        agg_blocks[kern] = {stat: mean_std(vals) for stat, vals in b_acc[kern].items()}
        bl = agg_blocks[kern]
        label = {"spectrum": f"Spectrum k={K} norm", "levenshtein": f"Levenshtein s={SIGMA}"}[kern]
        print(f"{label:<22}{bl['within_spam'][0]:>9.3f}±{bl['within_spam'][1]:<5.3f}"
              f"{bl['within_ham'][0]:>9.3f}±{bl['within_ham'][1]:<5.3f}"
              f"{bl['between'][0]:>9.3f}±{bl['between'][1]:<5.3f}"
              f"{bl['spam_over_between'][0]:>9.1f}±{bl['spam_over_between'][1]:<4.1f}")

    # persist aggregates + raw per-seed values
    with open("results/metrics.json", "w") as f:
        json.dump({
            "config": {"seeds": SEEDS, "figure_seed": FIGURE_SEED, "subsample": SUBSAMPLE,
                       "train_frac": TRAIN_FRAC, "k": K, "sigma": SIGMA, "C": C,
                       "majority_acc": [maj_m, maj_s]},
            "levenshtein_psd_figure_seed": d,
            "metrics_mean_std": agg_metrics,
            "ranking_mean_std": auc_summary,
            "class_blocks_mean_std": agg_blocks,
            "raw_per_seed": {"metrics": {n: dict(v) for n, v in m_acc.items()},
                             "support_vectors": dict(m_sv),
                             "class_blocks": {k: dict(v) for k, v in b_acc.items()}},
        }, f, indent=2)
    print("\nsaved results/metrics.json")
    print("saved figures/roc_pr_curves.png")

    # side-by-side kernel heatmaps from the representative split
    Ksp, Klv, n_ham = fig_grams
    plot_kernel_comparison(
        [(Ksp, f"Spectrum k={K} (normalized)", "cosine similarity"),
         (Klv, f"Levenshtein (sigma={SIGMA})", "RBF-edit similarity")],
        n_ham, "figures/kernel_heatmaps.png",
        suptitle=f"Spectrum isolates a spam cluster; Levenshtein washes it out "
                 f"(seed {FIGURE_SEED}, panels independently scaled 5-99th pct)")
    print("saved figures/kernel_heatmaps.png")


if __name__ == "__main__":
    main()