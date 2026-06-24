"""Smoke test for Project 8. Run from the project root:  python test_setup.py"""
import numpy as np
from collections import Counter

# B/C check 1: courselib must import (you vendored it into ./courselib)
from courselib.models.svm import BinaryKernelSVM
print("[1/5] courselib import...................... OK")

# B/C check 2: text-safe spectrum kernel — a CLASS with __call__(X1, X2) -> Gram matrix.
# Same interface as the course version, but counts only k-mers that appear,
# so a large text alphabet does NOT blow up.
class SpectrumKernel:
    def __init__(self, k=2):
        self.k = k
    def _counts(self, s):
        return Counter(s[i:i+self.k] for i in range(len(s) - self.k + 1))
    def __call__(self, X1, X2):
        c1 = [self._counts(s) for s in X1]
        c2 = [self._counts(s) for s in X2]
        K = np.zeros((len(X1), len(X2)))
        for i, a in enumerate(c1):
            for j, b in enumerate(c2):
                K[i, j] = sum(a[u] * b[u] for u in a.keys() & b.keys())
        return K

# C check 3: labels MUST be +1 / -1 for the SVM.
spam = ["win a free prize claim now", "free entry win cash prize now", "urgent claim your free cash"]
ham  = ["hey are we still on for lunch", "see you at the meeting later", "thanks for your help today"]
X = np.array(spam + ham)
Y = np.array([1, 1, 1, -1, -1, -1])
print("[2/5] custom string kernel defined.......... OK")

svm = BinaryKernelSVM(C=1.0, kernel="custom", kernel_function=SpectrumKernel(k=2))
svm.fit(X, Y)
print("[3/5] fit() with +/-1 labels................ OK")

pred = svm(X)
acc = np.mean(pred == Y) * 100
n_sv = len(svm.alphas)
print("[4/5] predict()............................. OK  ->", pred.tolist())
print(f"[5/5] training accuracy {acc:.0f}%, support vectors: {n_sv}")
assert acc >= 80, "Accuracy unexpectedly low — check labels/kernel."
print("\nALL CHECKS PASSED — courselib + custom string kernel + SVM all work.")