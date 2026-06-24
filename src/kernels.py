import numpy as np
from collections import Counter


class SpectrumKernel:
    """Spectrum (k-mer) kernel for text: counts shared length-k substrings.
    Dict-based -- only k-mers that actually occur -- so a large text alphabet
    does NOT blow up. Interface matches courselib kernels: __call__(X1, X2) -> Gram."""

    def __init__(self, k=3):
        self.k = k

    def _counts(self, s):
        return Counter(s[i:i + self.k] for i in range(len(s) - self.k + 1))

    def __call__(self, X1, X2):
        c1 = [self._counts(s) for s in X1]
        c2 = [self._counts(s) for s in X2]
        K = np.zeros((len(X1), len(X2)))
        for i, a in enumerate(c1):
            for j, b in enumerate(c2):
                K[i, j] = sum(a[u] * b[u] for u in a.keys() & b.keys())
        return K
    
    def normalized(self, X1, X2):
        """Cosine-normalized kernel: values in [0, 1], diagonal = 1.
        Removes the length bias of raw k-mer counts."""
        K = self(X1, X2)
        d1 = np.sqrt(np.array([self([s], [s])[0, 0] for s in X1]))
        d2 = np.sqrt(np.array([self([s], [s])[0, 0] for s in X2]))
        denom = np.outer(d1, d2)
        denom[denom == 0] = 1.0          # guard against empty strings
        return K / denom