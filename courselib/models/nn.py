import numpy as np
from .base import TrainableModel

class ReLU:
    """ReLU activation function and its subgradient."""

    def __call__(self, x):
        return np.maximum(0, x)

    def grad(self, x):
        return (x > 0).astype(float)

class Sigmoid:
    """Sigmoid activation function and its gradient."""

    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def grad(self, x):
        sig = self.__call__(x)
        return sig * (1 - sig)
    
class Linear:
    """Linear activation (identity function) and its derivative."""

    def __call__(self, x):
        return x

    def grad(self, x):
        return np.ones_like(x)
    
class Softmax:
    """Softmax activation function (row-wise) and its gradient."""

    def __call__(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))  # stability
        return exps / np.sum(exps, axis=1, keepdims=True)

    def grad(self, x):
        raise NotImplementedError("Gradient is handled with loss (e.g. cross-entropy)")
    
class MSE:
    """Quadratic (L2) loss and its gradient."""

    def __call__(self, Y_pred, Y_true):
        return 0.5 * np.mean((Y_pred - Y_true) ** 2)

    def grad(self, Y_pred, Y_true):
        return (Y_pred - Y_true) / Y_pred.shape[0]
    
class CrossEntropy:
    """Cross-entropy loss for one-hot targets and its gradient.
       Gradient for Softmax activation (to be used only together)."""

    def __call__(self, Y_pred, Y_true):
        return -np.mean(np.sum(Y_true * np.log(Y_pred), axis=1))

    def grad(self, Y_pred, Y_true):
        return (Y_pred - Y_true) / Y_pred.shape[0]
    

class DenseLayer:
    """
    Fully-connected (dense) layer with activation.
    
    Parameters:
    - num_in: number of input neurons
    - num_out: number of output neurons
    - activation: activation class (e.g., ReLU or Linear)
    - layer_name: unique name to identify the layer (used for parameter keys)
    """
    
    def __init__(self, num_in, num_out, activation='ReLU', layer_name=None):
        self.num_in = num_in
        self.num_out = num_out
        if activation == "ReLU":
            self.activation = ReLU()
        elif activation == "Sigmoid":
            self.activation = Sigmoid()
        elif activation == "Linear":
            self.activation = Linear()
        elif activation == "Softmax":
            self.activation = Softmax()
        elif activation == 'Custom':
            self.activation = activation()
        else:
            raise ValueError(f"Unknown activation: {activation}.")

        self.name = layer_name or f"Dense_{num_in}_{num_out}"

        # He initialization (good for ReLU)
        self.W = np.random.normal(loc=0.0, scale=np.sqrt(2. / num_in), size=(num_in, num_out))
        self.b = np.zeros((1, num_out))

    def __call__(self, X):
        """
        Forward pass through the layer.
        
        Parameters:
        - X: input matrix of shape (N, num_in)
        
        Returns:
        - z: pre-activation (N, num_out)
        - x: post-activation (N, num_out)
        """
        z = X @ self.W + self.b
        x = self.activation(z)
        return z, x

    def compute_delta(self, z, W_next, delta_next):
        """
        Compute backpropagated error for this layer.
        
        Parameters:
        - z: pre-activation from this layer (N, num_out)
        - W_next: weights of the next layer (num_out, next_layer_size)
        - delta_next: delta from the next layer (N, next_layer_size)
        
        Returns:
        - delta for this layer (N, num_out)
        """
        return (delta_next @ W_next.T) * self.activation.grad(z)

    def loss_grad(self, X_prev, delta):
        """
        Compute gradients w.r.t. weights and biases.
        
        Parameters:
        - X_prev: input to this layer (N, num_in)
        - delta: error signal from this layer (N, num_out)
        
        Returns:
        - Dictionary with gradients for weights and biases
        """
        w_grad = np.mean(delta[:, :, None] * X_prev[:, None, :], axis=0).T  # (num_in, num_out)
        b_grad = np.mean(delta, axis=0, keepdims=True)                      # (1, num_out)

        return {f'{self.name}_W': w_grad, f'{self.name}_b': b_grad}

    def _get_params(self):
        """Return a dictionary of parameters for this layer."""
        return {f'{self.name}_W': self.W, f'{self.name}_b': self.b}


class MLP(TrainableModel):
    def __init__(self, widths, optimizer, activation="ReLU", output_activation="Linear", loss="MSE"):
        """
        Initializes a multi-layer perceptron (MLP) using a sequence of DenseLayers.
        
        Parameters:
        - widths: list of layer sizes, including input and output dimensions
        - optimizer: optimizer instance (must support `update(params, grads)`)
        - activation: activation class for hidden layers
        - output_activation: activation class for the output layer
        - loss: loss function class
        """
        self.optimizer = optimizer
        self.widths = widths
        if loss == "MSE":
            self.loss = MSE()
        elif loss == "CE":
            self.loss = CrossEntropy()
        elif loss == 'Custom':
            self.loss = loss()
        else:
            raise ValueError("Unknown loss.")
        
        # Build hidden layers
        self.layers = [
            DenseLayer(widths[i], widths[i+1], activation=activation, layer_name=f"layer_{i}")
            for i in range(len(widths) - 2)
        ]
        
        # Output layer
        self.layers.append(
            DenseLayer(widths[-2], widths[-1], activation=output_activation, layer_name=f"layer_{len(widths)-2}")
        )
    
    def decision_function(self, X):
        """Applies all layers to compute the raw network output."""
        out = X
        for layer in self.layers:
            _, out = layer(out)
        return out
    
    def __call__(self, X):
        return np.argmax(self.decision_function(X), axis=-1)

    def forward_pass(self, X):
        """Computes pre-activations and activations at each layer."""
        x_l = [X]
        z_l = [X]
        for layer in self.layers:
            z, x = layer(x_l[-1])
            z_l.append(z)
            x_l.append(x)
        return z_l, x_l

    def backward_pass(self, X, Y, z_l, x_l):
        """Computes layer-wise gradients using backpropagation."""
        if isinstance(self.layers[-1].activation, Softmax)and isinstance(self.loss, CrossEntropy): # Softmax-cross-entropy case 
            delta = self.loss.grad(x_l[-1], Y)
        else:
            delta = self.loss.grad(x_l[-1], Y) * self.layers[-1].activation.grad(z_l[-1])
        deltas = [delta]
        
        for i in reversed(range(len(self.layers) - 1)):
            delta = self.layers[i].compute_delta(z_l[i+1], self.layers[i+1].W, deltas[-1])
            deltas.append(delta)
        
        return deltas[::-1]

    def loss_grad(self, X, Y):
        """Returns gradients of the loss w.r.t. all layer parameters."""
        z_l, x_l = self.forward_pass(X)
        delta_l = self.backward_pass(X, Y, z_l, x_l)

        grads = {}
        for i, layer in enumerate(self.layers):
            grads.update(layer.loss_grad(x_l[i], delta_l[i]))
        return grads

    def _get_params(self):
        """Returns a dictionary of all layer parameters."""
        params = {}
        for layer in self.layers:
            params.update(layer._get_params())
        return params