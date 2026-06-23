import numpy as np

def labels_encoding(Y, labels=None, pos_value=1, neg_value=-1):
    """
    Encodes class labels into a one-vs-rest style matrix with custom values.

    Parameters:
    - Y: array-like of shape (N,) – class labels
    - labels: optional list of label values in desired order; if None, inferred from sorted unique values
    - pos_value: value for the positive (true) class (default: 1)
    - neg_value: value for the negative class (default: -1)

    Returns:
    - encoded: ndarray of shape (N, K), where K = number of classes
    """
    Y = np.asarray(Y)
    if labels is None:
        labels = np.unique(Y)
    label_to_index = {label: i for i, label in enumerate(labels)}
    
    K = len(labels)
    N = len(Y)
    
    encoded = np.full((N, K), neg_value, dtype=float)
    for i, y in enumerate(Y):
        k = label_to_index[y]
        encoded[i, k] = pos_value

    return encoded

def labels_to_numbers(labels, class_names=None):
    if class_names is None:
        class_names = np.unique(labels)
    label_to_number = {label: i for i, label in enumerate(class_names)}
    return np.array([label_to_number[label] for label in labels])
