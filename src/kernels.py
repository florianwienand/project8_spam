import numpy as np
from collections import Counter
from scipy import sparse
from rapidfuzz.process import cdist
from rapidfuzz.distance import Levenshtein


class SpectrumKernel:
    """Spectrum (k-mer) kernel: K(s, t) is the dot product of the two k-mer
    count vectors. Built as a shared-vocabulary sparse count matrix plus one
    sparse matmul, so only k-mers that actually occur are tracked.
    Interface matches courselib kernels: __call__(X1, X2) -> Gram."""

    def __init__(self, k=3):
        self.k = k

    def _counts(self, s):
        return Counter(s[i:i + self.k] for i in range(len(s) - self.k + 1))

    def _matrix(self, X, vocab):
        """Sparse (n x |vocab|) k-mer count matrix over a fixed vocabulary."""
        indptr, indices, data = [0], [], []
        for s in X:
            for u, cnt in self._counts(s).items():
                indices.append(vocab[u])
                data.append(cnt)
            indptr.append(len(indices))
        return sparse.csr_matrix((data, indices, indptr),
                                 shape=(len(X), len(vocab)), dtype=float)

    def __call__(self, X1, X2):
        # shared vocabulary keeps the two count matrices column-aligned
        vocab = {}
        for X in (X1, X2):
            for s in X:
                for u in self._counts(s):
                    if u not in vocab:
                        vocab[u] = len(vocab)
        P1, P2 = self._matrix(X1, vocab), self._matrix(X2, vocab)
        return np.asarray((P1 @ P2.T).todense())

    def normalized(self, X1, X2):
        """Cosine-normalized kernel: values in [0, 1], diagonal = 1. Removes
        the length bias of raw k-mer counts."""
        K = self(X1, X2)
        d1 = np.sqrt(np.array([sum(c * c for c in self._counts(s).values()) for s in X1]))
        d2 = np.sqrt(np.array([sum(c * c for c in self._counts(s).values()) for s in X2]))
        denom = np.outer(d1, d2)
        denom[denom == 0] = 1.0          # strings shorter than k
        return K / denom


class LevenshteinKernel:
    """RBF kernel over length-normalized edit distance,

        K(s, t) = exp(-d(s, t)^2 / (2 * sigma^2)),   d in [0, 1].

    Not PSD in general, so the training Gram can be indefinite and break
    cvxopt's QP solver. psd_fix repairs the square training Gram only:

      None    -- raw kernel, used to inspect the eigenvalues or for the kPCA route
      "shift" -- add rho*I with rho = -lambda_min; touches only the diagonal,
                 every off-diagonal similarity stays identical to the raw kernel
      "clip"  -- eigenvalue clipping; also changes off-diagonals, so train and
                 test kernels no longer match (kept for comparison only)

    The repair triggers on `X1 is X2`: BinaryKernelSVM.fit calls kernel(X, X)
    with the same array, decision_function with different arrays (stays raw)."""

    def __init__(self, sigma=0.3, psd_fix="shift"):
        self.sigma = sigma
        self.psd_fix = psd_fix

    def _raw(self, X1, X2):
        D = cdist(list(X1), list(X2),
                  scorer=Levenshtein.normalized_distance, workers=-1)
        return np.exp(-(D ** 2) / (2 * self.sigma ** 2))

    def __call__(self, X1, X2):
        K = self._raw(X1, X2)

        if not self.psd_fix or X1 is not X2:
            return K

        K = (K + K.T) / 2.0                      # remove tiny float asymmetry

        if self.psd_fix == "shift":
            lam_min = np.linalg.eigvalsh(K)[0]   # ascending; smallest first
            if lam_min < 0:
                K = K + (-lam_min + 1e-8) * np.eye(len(K))
            return K

        if self.psd_fix == "clip":
            w, V = np.linalg.eigh(K)
            w = np.clip(w, 0, None)
            return (V * w) @ V.T

        raise ValueError(f"unknown psd_fix: {self.psd_fix!r} (use None, 'shift', or 'clip')")

    def diagnostics(self, X):
        """PSD report for the raw training Gram: eigenvalue range, negative
        mass, and how much each repair perturbs the matrix (relative Frobenius
        norm; shift only moves the diagonal, clip also moves off-diagonals)."""
        K = self._raw(X, X)
        K = (K + K.T) / 2.0
        w = np.linalg.eigvalsh(K)
        fro = np.linalg.norm(K)
        n = len(K)
        neg = w[w < 0]
        rho = float(max(0.0, -w[0]))
        return {
            "n": n,
            "lambda_min": float(w[0]),
            "lambda_max": float(w[-1]),
            "n_negative": int(neg.size),
            "neg_eig_mass": float(np.sum(np.abs(neg)) / np.sum(np.abs(w))),
            "shift_rho": rho,
            "shift_rel_perturbation": float(rho * np.sqrt(n) / fro),
            "clip_rel_perturbation": float(np.sqrt(np.sum(neg ** 2)) / fro),
        }


def levenshtein_kpca_features(X_train, X_test, sigma=0.45, var_keep=0.99):
    """Project the indefinite Levenshtein kernel onto its positive
    eigen-spectrum (kernel PCA) and return explicit features for a linear SVM.
    Keeps the top positive eigen-directions covering `var_keep` of the positive
    eigenvalue mass; test points are projected with the Nystrom formula, so
    train and test stay consistent and the diagonal is never inflated (the
    failure mode of the shift repair).

    Returns (Phi_train, Phi_test) for BinaryKernelSVM(kernel='linear')."""
    raw = LevenshteinKernel(sigma=sigma, psd_fix=None)
    K = raw(X_train, X_train)
    K = (K + K.T) / 2.0
    w, V = np.linalg.eigh(K)

    pos = w > 1e-8
    w, V = w[pos], V[:, pos]
    order = np.argsort(w)[::-1]          
    w, V = w[order], V[:, order]

    cum = np.cumsum(w) / w.sum()
    keep = int(np.searchsorted(cum, var_keep)) + 1
    keep = max(1, min(keep, len(w)))
    w, V = w[:keep], V[:, :keep]

    Phi_train = V * np.sqrt(w)                       # (n_train, keep)
    Phi_test = raw(X_test, X_train) @ V / np.sqrt(w)  # Nystrom projection
    return Phi_train, Phi_test