"""Project 8 -- MVP: spectrum-kernel SVM on SMS spam.
Run from the project root:  python run_experiments.py"""
import numpy as np

from src.preprocessing import load_sms, clean_text, make_split
from src.kernels import SpectrumKernel
from src.evaluation import binary_metrics, plot_kernel_heatmap
from courselib.models.svm import BinaryKernelSVM

# --- config (no magic numbers buried in the logic) ---
SEED = 0
DATA_PATH = "data/SMSSpamCollection"
SUBSAMPLE = 1000        # cvxopt stays fast here; raise later and watch the timing
TRAIN_FRAC = 0.75
K = 3                   # spectrum kernel order
C = 1.0                 # SVM soft-margin


def main():
    X_raw, y = load_sms(DATA_PATH)
    X = np.array([clean_text(t) for t in X_raw], dtype=object)
    print(f"loaded {len(X)} messages | spam={np.sum(y==1)} ham={np.sum(y==-1)}")

    X_tr, y_tr, X_te, y_te = make_split(X, y, SUBSAMPLE, TRAIN_FRAC, seed=SEED)
    print(f"subsample={len(X_tr)+len(X_te)} | train={len(X_tr)} test={len(X_te)}")

    svm = BinaryKernelSVM(C=C, kernel="custom", kernel_function=SpectrumKernel(k=K))
    svm.fit(X_tr, y_tr)
    pred = svm(X_te)

    m = binary_metrics(pred, y_te)
    print(f"\naccuracy={m['accuracy']:.1f}%  precision={m['precision']:.2f}  "
          f"recall={m['recall']:.2f}  #SV={len(svm.alphas)}")
    print("confusion:", m["confusion"])

    k3 = SpectrumKernel(k=K)
    plot_kernel_heatmap(k3.normalized, X_tr, y_tr,
                        "figures/spectrum_heatmap_norm.png",
                        title=f"Normalized spectrum kernel (k={K}), sorted by class")
    print("saved figures/spectrum_heatmap.png")


if __name__ == "__main__":
    main()