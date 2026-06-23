"""Plug the string kernels into courselib's kernel SVM."""
from courselib.models.svm import BinaryKernelSVM


def make_kernel_svm(kernel_fn, C=1.0):
    """kernel_fn(X1, X2) -> Gram matrix. courselib accepts a custom kernel.
    Fix n / sigma with functools.partial before passing kernel_fn here."""
    return BinaryKernelSVM(C=C, kernel="custom", kernel_function=kernel_fn)