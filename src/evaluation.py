import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def binary_metrics(pred, y_true):
    """Accuracy / precision / recall for +1/-1 labels (+1 = spam = positive)."""
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == -1)))
    fn = int(np.sum((pred == -1) & (y_true == 1)))
    tn = int(np.sum((pred == -1) & (y_true == -1)))
    acc = (tp + tn) / len(y_true) * 100
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec,
            "confusion": {"TP": tp, "FP": fp, "FN": fn, "TN": tn}}


def class_block_means(K, y):
    """Mean off-diagonal kernel value within-spam, within-ham, and between
    classes (+1 = spam). A spam/between ratio of 1.0 means no clustering."""
    m = ~np.eye(len(y), dtype=bool)
    spam = (y[:, None] == 1) & (y[None, :] == 1) & m
    ham = (y[:, None] == -1) & (y[None, :] == -1) & m
    btw = (y[:, None] != y[None, :])
    ws, wh, b = float(K[spam].mean()), float(K[ham].mean()), float(K[btw].mean())
    return {"within_spam": ws, "within_ham": wh, "between": b,
            "spam_over_between": ws / b if b else float("nan")}


def plot_kernel_comparison(panels, n_ham, path,
                           suptitle="Kernel matrices sorted by class (ham, then spam)"):
    """Side-by-side kernel heatmaps with the spam-spam block outlined.

    panels : list of (K_sorted, title, colorbar_label); Grams already sorted by
             class (ham first). Each panel is scaled to its own 5-99th
             percentile of off-diagonal values.
    n_ham  : number of ham rows, i.e. where the spam block starts.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(6.3 * len(panels), 5.4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (K, title, clabel) in zip(axes, panels):
        n = len(K)
        off = K[~np.eye(n, dtype=bool)]
        vmin, vmax = np.percentile(off, 5), np.percentile(off, 99)
        im = ax.imshow(K, cmap="viridis", vmin=vmin, vmax=vmax,
                       aspect="equal", interpolation="nearest")
        ax.axhline(n_ham, color="white", lw=0.8, ls="--", alpha=0.7)
        ax.axvline(n_ham, color="white", lw=0.8, ls="--", alpha=0.7)
        ax.add_patch(Rectangle((n_ham, n_ham), n - n_ham, n - n_ham,
                               fill=False, edgecolor="white", lw=1.4))
        ax.text(n - 2, n_ham - 8, "spam-spam block", color="white",
                ha="right", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("message (sorted: ham then spam)", fontsize=8)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(clabel, fontsize=8)
    fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()


def plot_kernel_heatmap(kernel, X, y, path, title="Kernel matrix (sorted by class)", vmax=None):
    """Single kernel heatmap, rows/cols sorted by class (superseded by
    plot_kernel_comparison)."""
    order = np.argsort(y)
    K = kernel(X[order], X[order])
    if vmax is None:
        off = K[~np.eye(len(K), dtype=bool)]
        vmax = np.percentile(off, 99)
    plt.figure(figsize=(5, 4))
    plt.imshow(K, cmap="viridis", aspect="auto", vmin=0, vmax=vmax)
    plt.title(title)
    plt.colorbar(label=f"similarity (capped at {vmax:.2f})")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()

def plot_roc_pr(model_scores, path, prevalence=None,
                suptitle="ROC and precision-recall (mean over seeds, band = +/-1 std)"):
    """ROC and precision-recall curves averaged over seeds. Both sweep the
    decision threshold, so they show ranking quality independent of the
    default cutoff.

    model_scores : dict name -> list of (scores, y_true) per seed, +1 = spam.
    prevalence   : spam fraction, drawn as the no-skill line on the PR panel.

    Returns {name: {"roc_auc": [mean, std], "avg_precision": [mean, std]}}.
    """
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    fpr_grid = np.linspace(0, 1, 200)
    rec_grid = np.linspace(0, 1, 200)
    fig, (axr, axp) = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    summary = {}

    for ci, (name, seeds) in enumerate(model_scores.items()):
        tprs, aucs, precs, aps = [], [], [], []
        for scores, y in seeds:
            yb = (np.asarray(y) == 1).astype(int)
            s = np.asarray(scores)
            fpr, tpr, _ = roc_curve(yb, s)
            it = np.interp(fpr_grid, fpr, tpr); it[0] = 0.0
            tprs.append(it); aucs.append(auc(fpr, tpr))
            p, r, _ = precision_recall_curve(yb, s)
            # interpolated precision: max precision achieved at recall >= grid point
            ip = np.array([p[r >= rg].max() if np.any(r >= rg) else 0.0 for rg in rec_grid])
            precs.append(ip); aps.append(average_precision_score(yb, s))
        c = colors[ci]
        tpr_m, tpr_s = np.mean(tprs, 0), np.std(tprs, 0)
        pr_m, pr_s = np.mean(precs, 0), np.std(precs, 0)
        axr.plot(fpr_grid, tpr_m, color=c, lw=1.6, label=f"{name}  (AUC {np.mean(aucs):.3f})")
        axr.fill_between(fpr_grid, np.clip(tpr_m - tpr_s, 0, 1), np.clip(tpr_m + tpr_s, 0, 1),
                         color=c, alpha=0.12)
        axp.plot(rec_grid, pr_m, color=c, lw=1.6, label=f"{name}  (AP {np.mean(aps):.3f})")
        axp.fill_between(rec_grid, np.clip(pr_m - pr_s, 0, 1), np.clip(pr_m + pr_s, 0, 1),
                         color=c, alpha=0.12)
        summary[name] = {"roc_auc": [float(np.mean(aucs)), float(np.std(aucs))],
                         "avg_precision": [float(np.mean(aps)), float(np.std(aps))]}

    axr.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="chance")
    axr.set(xlabel="false positive rate", ylabel="true positive rate (recall)",
            title="ROC", xlim=(0, 1), ylim=(0, 1.02))
    axr.legend(fontsize=7, loc="lower right")
    if prevalence is not None:
        axp.axhline(prevalence, color="k", ls="--", lw=0.8, alpha=0.5,
                    label=f"no-skill ({prevalence:.2f})")
    axp.set(xlabel="recall", ylabel="precision", title="Precision-Recall",
            xlim=(0, 1), ylim=(0, 1.02))
    axp.legend(fontsize=7, loc="lower left")
    fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    return summary


def plot_obfuscation_robustness(levels, curves, path,
                                suptitle="Spam recall under adversarial obfuscation "
                                         "(mean over seeds, band = +/-1 std)"):
    """Recall vs obfuscation intensity, one line per representation.
    curves : dict name -> array (n_seeds x n_levels) of recall values."""
    levels = np.asarray(levels)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for ci, (name, R) in enumerate(curves.items()):
        R = np.asarray(R)
        m, s = R.mean(0), R.std(0)
        c = colors[ci]
        ax.plot(levels, m, "-o", color=c, lw=1.8, ms=4, label=name)
        ax.fill_between(levels, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), color=c, alpha=0.15)
    ax.set(xlabel="obfuscation level  (leetspeak + intra-word spacing)",
           ylabel="spam recall", ylim=(0, 1.02))
    ax.set_title(suptitle, fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()