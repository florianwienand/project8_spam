import numpy as np

def standardize(x):
    """
    Standardization normalization.
    Scales the data to have mean 0 and standard deviation 1.
    Parameters:
    - x: numpy array of shape (n_samples, n_features)
    Returns:
    - numpy array of the same shape as x, with mean 0 and std 1
    """
    return (x - np.mean(x, axis=0))/np.std(x,axis=0)


def min_max(x):
    """
    Min-Max normalization.
    Scales the data to a range of [0, 1] by transforming each feature individually.
    Parameters:
    - x: numpy array of shape (n_samples, n_features)
    Returns:
    - numpy array of the same shape as x, with all values scaled to the range [0, 1]
    """
    return (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))