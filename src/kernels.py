"""String kernels: spectrum (n-gram) and Levenshtein edit-distance.
Each returns a precomputed Gram matrix to feed into a kernel SVM."""
import numpy as np


def spectrum_kernel_matrix(X1, X2, n=3):
    """Count shared length-n substrings. Shape (len(X1), len(X2)). TODO."""
    raise NotImplementedError


def levenshtein_kernel_matrix(X1, X2, sigma=1.0):
    """K = exp(-d^2 / (2*sigma^2)) from edit distance. Shape (len(X1), len(X2)).
    TODO: implement (use rapidfuzz for fast edit distance)."""
    raise NotImplementedError