import numpy as np
from collections import Counter
from scipy import sparse
from rapidfuzz.process import cdist
from rapidfuzz.distance import Levenshtein


class SpectrumKernel:
    """Spectrum (k-mer) kernel for text: counts shared length-k substrings.
    Only k-mers that actually occur are tracked, so a large text alphabet does NOT
    blow up. K(s, t) is the dot product of their k-mer count vectors -- exactly what
    the naive nested loop computes, but we assemble a shared-vocabulary sparse count
    matrix and read the whole Gram off a single sparse matmul, which is orders of
    magnitude faster on real text. Interface matches courselib kernels:
    __call__(X1, X2) -> Gram."""

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
        # shared vocabulary; only k-mers present in BOTH sets can move the dot product,
        # but indexing everything keeps the two count matrices column-aligned.
        vocab = {}
        for X in (X1, X2):
            for s in X:
                for u in self._counts(s):
                    if u not in vocab:
                        vocab[u] = len(vocab)
        P1, P2 = self._matrix(X1, vocab), self._matrix(X2, vocab)
        return np.asarray((P1 @ P2.T).todense())

    def normalized(self, X1, X2):
        """Cosine-normalized kernel: values in [0, 1], diagonal = 1 for non-empty strings.
        Removes the length bias of raw k-mer counts. Self-similarity K(s, s) is just the
        sum of squared k-mer counts, so we get the norms without a second Gram."""
        K = self(X1, X2)
        d1 = np.sqrt(np.array([sum(c * c for c in self._counts(s).values()) for s in X1]))
        d2 = np.sqrt(np.array([sum(c * c for c in self._counts(s).values()) for s in X2]))
        denom = np.outer(d1, d2)
        denom[denom == 0] = 1.0          # guard against strings shorter than k
        return K / denom


class LevenshteinKernel:
    """RBF kernel over length-normalized edit distance:

        K(s, t) = exp(-d(s, t)^2 / (2 * sigma^2)),   d = normalized Levenshtein in [0, 1].

    The edit-distance RBF is NOT positive semi-definite in general, so the training
    Gram matrix can be indefinite and break cvxopt's QP solver. The repair below keeps
    training and prediction consistent -- they use the SAME similarities.

    psd_fix:
      None      -- raw kernel, untouched. Use it to inspect indefiniteness (eigenvalues)
                   or to run a Krein / indefinite-kernel SVM where train AND test both
                   use the raw kernel (consistent by construction, QP may be non-convex).

      "shift"   -- (default) diagonal shift. Add rho*I to the training Gram with
                   rho = -lambda_min, lifting the smallest eigenvalue to ~0 so cvxopt
                   sees a PSD matrix. Because the identity is diagonal in every basis,
                   this changes ONLY the self-similarity (diagonal) entries and leaves
                   every off-diagonal -- i.e. every similarity actually used at prediction
                   time -- byte-for-byte identical to the raw kernel. Equivalent to adding
                   rho to every eigenvalue (K + rho*I = V (Lambda + rho) V^T). This is the
                   standard, consistent way to regularize an indefinite SVM kernel; the
                   diagonal term just acts as regularization during the QP and is absent
                   at test time (test points are distinct from support vectors).

      "clip"    -- eigenvalue clipping: rebuild K = V * clip(Lambda, 0) * V^T. This also
                   changes the OFF-diagonal entries, so the kernel used in training no
                   longer matches the raw kernel used at prediction -> train/test
                   inconsistency. Kept ONLY so the inconsistency can be measured against
                   "shift"; do not report final numbers with it.

    The repair runs only on the square training Gram, detected by `X1 is X2`
    (courselib's BinaryKernelSVM.fit calls self.kernel(X, X) with the same array;
    decision_function calls self.kernel(X_test, sv) with different arrays -> raw)."""

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
        """PSD report for the RAW training Gram on X -- the numbers to cite when defending
        the kernel. Reports eigenvalue range, how indefinite the kernel is, and how much
        each repair perturbs the matrix (Frobenius norm, relative to the raw kernel)."""
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
            # share of the spectrum's total magnitude that is negative (0 = PSD, larger = more indefinite)
            "neg_eig_mass": float(np.sum(np.abs(neg)) / np.sum(np.abs(w))),
            "shift_rho": rho,
            # ||rho*I||_F / ||K||_F  -- shift only moves the diagonal
            "shift_rel_perturbation": float(rho * np.sqrt(n) / fro),
            # ||K_clip - K||_F / ||K||_F  -- clip also moves off-diagonals
            "clip_rel_perturbation": float(np.sqrt(np.sum(neg ** 2)) / fro),
        }


def levenshtein_kpca_features(X_train, X_test, sigma=0.45, var_keep=0.99):
    """Use the indefinite Levenshtein kernel the RIGHT way.

    The Levenshtein RBF kernel is not PSD, and there is no clean way to feed it
    straight to the dual SVM solver:
      * raw            -> cvxopt rejects the indefinite QP and fails to fit;
      * diagonal shift -> makes it PSD but the shift (rho ~ -lambda_min) is large
                          relative to the small off-diagonals, so the training Gram
                          becomes strongly diagonally dominant and the SVM
                          over-regularizes into a degenerate (majority-class) classifier;
      * eigen-clip     -> changes off-diagonals, so train and test use different kernels.

    Instead we project onto the kernel's positive eigen-spectrum (kernel PCA): drop the
    indefinite directions, keep the top positive ones, and get EXPLICIT features. Train a
    plain linear SVM on those. The projection is consistent train<->test via the Nystrom
    out-of-sample formula and never inflates the diagonal, so it cannot over-regularize.

    Parameters
    ----------
    sigma : float          Levenshtein RBF width.
    var_keep : float       keep the top positive eigen-directions covering this fraction
                           of positive eigenvalue mass (0.99 = 99%).

    Returns
    -------
    (Phi_train, Phi_test)  dense feature matrices for BinaryKernelSVM(kernel='linear').
    """
    raw = LevenshteinKernel(sigma=sigma, psd_fix=None)
    K = raw(X_train, X_train)
    K = (K + K.T) / 2.0
    w, V = np.linalg.eigh(K)

    pos = w > 1e-8                       # keep strictly-positive eigen-directions only
    w, V = w[pos], V[:, pos]
    order = np.argsort(w)[::-1]          
    w, V = w[order], V[:, order]

    cum = np.cumsum(w) / w.sum()
    keep = int(np.searchsorted(cum, var_keep)) + 1
    keep = max(1, min(keep, len(w)))
    w, V = w[:keep], V[:, :keep]

    Phi_train = V * np.sqrt(w)                       # (n_train, keep)
    Phi_test = raw(X_test, X_train) @ V / np.sqrt(w)  # Nystrom projection -> (n_test, keep)
    return Phi_train, Phi_test