"""
Project 8: Spam Detection with String Kernels.
One-command reproduction. Run from the project root:

    python run_experiments.py

TODO: load -> preprocess -> build kernels -> train SVMs -> evaluate -> save outputs.
"""
import numpy as np

from courselib.models.svm import BinaryKernelSVM          # supports kernel='custom'
from src.preprocessing import clean_text
from src.kernels import spectrum_kernel_matrix, levenshtein_kernel_matrix
from src.evaluation import evaluate_model

SEED = 0


def main():
    np.random.seed(SEED)
    print("TODO: implement the experiment pipeline.")


if __name__ == "__main__":
    main()