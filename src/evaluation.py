import numpy as np
import matplotlib.pyplot as plt


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


def plot_kernel_heatmap(kernel, X, y, path, title="Kernel matrix (sorted by class)"):
    """Save a heatmap of kernel(X, X) with rows/cols sorted by class."""
    order = np.argsort(y)
    K = kernel(X[order], X[order])
    plt.figure(figsize=(5, 4))
    plt.imshow(K, cmap="viridis", aspect="auto")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()