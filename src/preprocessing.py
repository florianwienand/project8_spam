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
    """Lowercase + collapse whitespace. Digits and punctuation are kept on
    purpose, they carry spam signal for character-level kernels."""
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
    """Stratified subsample to `size`, then stratified train/test split.
    Indices are shuffled so train/test are not class-ordered."""
    rng = np.random.default_rng(seed)
    sub = stratified_indices(y, size, rng)
    rng.shuffle(sub)
    Xs, ys = X[sub], y[sub]
    tr = stratified_indices(ys, int(train_frac * len(ys)), rng)
    mask = np.zeros(len(ys), dtype=bool)
    mask[tr] = True
    return Xs[mask], ys[mask], Xs[~mask], ys[~mask]


def tfidf_features(X_train, X_test, ngram_range=(1, 2), min_df=2):
    """Word TF-IDF features for the baseline. Fit on the training split only,
    then transform both splits with that vocabulary (no leakage). Rows are
    L2-normalized by default, so the linear-kernel diagonal is 1, the same
    scale as the normalized string kernels.

    Returns (X_train_tfidf, X_test_tfidf, vectorizer)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df)
    X_train_tfidf = vec.fit_transform(list(X_train)).toarray()
    X_test_tfidf = vec.transform(list(X_test)).toarray()
    return X_train_tfidf, X_test_tfidf, vec

_LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7", "g": "9", "b": "8", "l": "1"}


def obfuscate(text, level, rng):
    """Spammer-style evasion at intensity `level` in [0, 1]: leetspeak
    substitution (each eligible character with probability `level`) and
    intra-word space insertion (probability 0.25*level per character)."""
    out = []
    for ch in text:
        low = ch.lower()
        out.append(_LEET[low] if (low in _LEET and rng.random() < level) else ch)
        if ch != " " and rng.random() < 0.25 * level:
            out.append(" ")
    return "".join(out)