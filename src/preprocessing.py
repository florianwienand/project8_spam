import csv
import numpy as np
import pandas as pd


def load_sms(path):
    """Load SMS Spam Collection (tab-separated: label<TAB>text).
    Returns X (texts) and y (+1 = spam, -1 = ham)."""
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"],
                     encoding="latin-1", quoting=csv.QUOTE_NONE, on_bad_lines="skip")
    X = df["text"].astype(str).to_numpy()
    y = np.where(df["label"].str.strip() == "spam", 1, -1)
    return X, y


def clean_text(text):
    """Minimal cleaning: lowercase + collapse whitespace. Digits and punctuation
    are kept on purpose -- they carry spam signal for character-level kernels."""
    return " ".join(str(text).lower().split())


def stratified_indices(y, n, rng):
    """Pick ~n indices while preserving the class ratio of y."""
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        k = round(n * len(ci) / len(y))
        idx.append(rng.choice(ci, size=min(k, len(ci)), replace=False))
    return np.concatenate(idx)


def make_split(X, y, size, train_frac, seed=0):
    """Stratified subsample to `size`, then stratified train/test split."""
    rng = np.random.default_rng(seed)
    sub = stratified_indices(y, size, rng)
    Xs, ys = X[sub], y[sub]
    tr = stratified_indices(ys, int(train_frac * len(ys)), rng)
    mask = np.zeros(len(ys), dtype=bool)
    mask[tr] = True
    return Xs[mask], ys[mask], Xs[~mask], ys[~mask]