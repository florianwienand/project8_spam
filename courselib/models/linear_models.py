import numpy as np
from .base import TrainableModel


class LinearRegression(TrainableModel):
    """Linear regression model."""

    def __init__(self, w, b, optimizer):
        super().__init__(optimizer)
        self.w = np.array(w, dtype=float)
        self.b = np.array(b, dtype=float)

    def loss_grad(self, X, y):
        residual = self.decision_function(X) - y
        grad_w = X.T @ residual / len(X)
        grad_b = np.mean(residual)
        return {"w": grad_w, "b": grad_b}
    
    def decision_function(self, X):
        return X @ self.w + self.b
    
    def _get_params(self):
        return {"w": self.w, "b": self.b}

    def __call__(self, X):
        return self.decision_function(X)
    

class LinearBinaryClassification(TrainableModel):
    """Linear binary classification model."""

    def __init__(self, w, b, optimizer, class_labels=[-1,1]):
        super().__init__(optimizer)
        self.w = np.array(w, dtype=float)
        self.b = np.array(b, dtype=float)
        self.class_labels = [min(class_labels),max(class_labels)]
        self.threshold = self.class_labels[0] + (self.class_labels[1] - self.class_labels[0])/2.

    def loss_grad(self, X, y):
        residual = self.decision_function(X) - y
        grad_w = X.T @ residual / len(X)
        grad_b = np.mean(residual)
        return {"w": grad_w, "b": grad_b}
    
    def decision_function(self, X):
        return X @ self.w + self.b
    
    def _get_params(self):
        return {"w": self.w, "b": self.b}

    def __call__(self, X):
        return np.where(self.decision_function(X)>=self.threshold, self.class_labels[1], self.class_labels[0])
    

class RidgeClassifier(LinearBinaryClassification):
    """
    Ridge binary classifier
    """
    def __init__(self, w, b, optimizer, lam=0.1, class_labels=[-1,1]):
        super().__init__(w, b, optimizer, class_labels)
        self.lam = lam

    def loss_grad(self, X,Y):
        """Loss gradient"""
        residual = self.decision_function(X) - Y
        grad_w = X.T@residual/X.shape[0] + 2*self.lam*self.w 
        grad_b = np.mean(residual, axis=0)
        return {"w": grad_w, "b": grad_b}
    

class LassoClassifier(LinearBinaryClassification):
    """
    Lasso binary classifier
    """
    def __init__(self, w, b, optimizer, lam=0.1, class_labels=[-1,1]):
        super().__init__(w, b, optimizer, class_labels)
        self.lam = lam

    def loss_grad(self, X,Y):
        """Loss gradient"""
        residual = self.decision_function(X) - Y
        grad_w = X.T@residual/X.shape[0] + self.lam*np.sign(self.w)
        grad_b = np.mean(residual, axis=0)
        return {"w": grad_w, "b": grad_b}


class LinearMulticlassClassification(TrainableModel):
    """
    Linear multiclass classifier with least-squares multioutput loss.
    Trained with gradient-based optimization.
    
    Parameters:
    - w: initial weight matrix, shape (d, K)
    - b: initial bias vector, shape (K,) or (1, K)
    - optimizer: an instance of Optimizer
    """

    def __init__(self, w, b, optimizer):
        super().__init__(optimizer)
        self.w = np.array(w, dtype=float)        # shape (d, K)
        self.b = np.array(b, dtype=float).reshape(1, -1)  # shape (1, K)

    def decision_function(self, X):
        """
        Computes class scores for input features X.

        Parameters:
        - X: input data, shape (N, d)

        Returns:
        - scores: array of shape (N, K)
        """
        return X @ self.w + self.b

    def loss_grad(self, X, Y):
        """
        Computes gradients of squared loss w.r.t. weights and biases.

        Parameters:
        - X: input features, shape (N, d)
        - Y: target matrix, shape (N, K), typically {-1, 1} encoding

        Returns:
        - gradients: dict with keys 'w' and 'b'
        """
        residual = self.decision_function(X) - Y           # (N, K)
        grad_w = X.T @ residual / len(X)                   # (d, K)
        grad_b = np.mean(residual, axis=0, keepdims=True)  # (1, K)

        return {"w": grad_w, "b": grad_b}

    def _get_params(self):
        """Returns current parameters as a dictionary."""
        return {"w": self.w, "b": self.b}

    def __call__(self, X):
        """
        Predicts class labels using the decision function.

        Parameters:
        - X: input features, shape (N, d)

        Returns:
        - labels: predicted class indices, shape (N,)
        """
        return np.argmax(self.decision_function(X), axis=-1)

    