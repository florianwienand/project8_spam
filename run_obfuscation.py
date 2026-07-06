"""Obfuscation robustness: train every representation on clean data, then
obfuscate only the spam test messages (leetspeak + intra-word spacing) at
rising intensity and track spam recall. Ham is never touched, so precision
cannot drift and every recall drop is a real detection loss.

run from the project root:  python run_obfuscation.py"""
import json
import numpy as np

from src.preprocessing import load_sms, clean_text, make_split, tfidf_features, obfuscate
from src.kernels import SpectrumKernel, levenshtein_kpca_features
from src.evaluation import plot_obfuscation_robustness
from courselib.models.svm import BinaryKernelSVM

SEEDS = [0, 1, 2, 3, 4]
DATA_PATH = "data/SMSSpamCollection"
SUBSAMPLE = 1000
TRAIN_FRAC = 0.75
SIGMA = 0.45
C = 1.0
LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
MODELS = ["word TF-IDF", "char spectrum k=3", "edit Levenshtein kPCA"]


def recall_precision(pred, y):
    tp = np.sum((pred == 1) & (y == 1)); fn = np.sum((pred == -1) & (y == 1))
    fp = np.sum((pred == 1) & (y == -1))
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    return rec, prec


def main():
    X_raw, y = load_sms(DATA_PATH)
    X = np.array([clean_text(t) for t in X_raw], dtype=object)
    print(f"obfuscation experiment | {len(SEEDS)} seeds | levels {LEVELS}\n")

    rec = {m: np.zeros((len(SEEDS), len(LEVELS))) for m in MODELS}
    prec = {m: np.zeros((len(SEEDS), len(LEVELS))) for m in MODELS}

    for si, seed in enumerate(SEEDS):
        X_tr, y_tr, X_te, y_te = make_split(X, y, SUBSAMPLE, TRAIN_FRAC, seed=seed)

        # level 0.0 leaves the text untouched, so it should match the clean
        # recall from run_experiments (consistency check between the scripts)
        _, _, vec = tfidf_features(X_tr, X_te)
        word = BinaryKernelSVM(C=C, kernel="linear")
        word.fit(vec.transform(list(X_tr)).toarray(), y_tr)
        char = BinaryKernelSVM(C=C, kernel="custom", kernel_function=SpectrumKernel(k=3).normalized)
        char.fit(X_tr, y_tr)
        # only the train features are needed here; the X_tr[:1] dummy arg is discarded
        Phi_tr, _ = levenshtein_kpca_features(X_tr, X_tr[:1], sigma=SIGMA)
        edit = BinaryKernelSVM(C=C, kernel="linear")
        edit.fit(Phi_tr, y_tr)

        rng = np.random.default_rng(10_000 + seed)
        spam_idx = np.where(y_te == 1)[0]
        for li, level in enumerate(LEVELS):
            Xo = X_te.copy()
            for i in spam_idx:
                Xo[i] = obfuscate(X_te[i], level, rng)
            r, p = recall_precision(word(vec.transform(list(Xo)).toarray()), y_te)
            rec["word TF-IDF"][si, li], prec["word TF-IDF"][si, li] = r, p
            r, p = recall_precision(char(Xo), y_te)
            rec["char spectrum k=3"][si, li], prec["char spectrum k=3"][si, li] = r, p
            _, Phi_o = levenshtein_kpca_features(X_tr, Xo, sigma=SIGMA)
            r, p = recall_precision(edit(Phi_o), y_te)
            rec["edit Levenshtein kPCA"][si, li], prec["edit Levenshtein kPCA"][si, li] = r, p
        print(f"  seed {seed} done")

    print("\nspam recall vs obfuscation level (mean over seeds):")
    print(f"{'model':<26}" + "".join(f"L={L:<6.1f}" for L in LEVELS))
    for m in MODELS:
        print(f"{m:<26}" + "".join(f"{rec[m][:, li].mean():<8.2f}" for li in range(len(LEVELS))))
    print("\nprecision (mean over all levels, stays high -> recall changes are real): "
          + ", ".join(f"{m.split()[0]} {prec[m].mean():.2f}" for m in MODELS))

    plot_obfuscation_robustness(LEVELS, rec, "figures/obfuscation_robustness.png")
    print("saved figures/obfuscation_robustness.png")

    with open("results/obfuscation.json", "w") as f:
        json.dump({"seeds": SEEDS, "levels": LEVELS,
                   "recall_mean": {m: rec[m].mean(0).tolist() for m in MODELS},
                   "recall_std": {m: rec[m].std(0).tolist() for m in MODELS},
                   "precision_mean": {m: prec[m].mean(0).tolist() for m in MODELS}}, f, indent=2)
    print("saved results/obfuscation.json")


if __name__ == "__main__":
    main()