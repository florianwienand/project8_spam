import numpy as np
    
class TrainableModel:
    """
    Base class for models trained using iterative optimization.
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def loss_grad(self, X, y):
        """Subclasses must override this to return parameter gradients."""
        raise NotImplementedError
    
    def decision_function(self, X):
        """Subclasses must override this to return model's decision function."""
        raise NotImplementedError
    
    def _get_params(self):
        """Subclasses must override this to return dictionary of model's parameters."""
        raise NotImplementedError

    def compute_metrics(self, X, Y, metrics_dict):
        """
        Compute metrics 

        Parameters:
        ----------
        X: array
          Features matrix 
        Y: array
          Labels vector
        metrics_dict: dict
          Dictionary with metric names as keys and
          functions to comoute metrics as values

          The expected syntax of a metric function is metric(y_pred, y_true) 
        
        Returns:
        -------
        metrics: dict
          Dictionary of current metric values
        """
        metrics = {}
        for metric_name in metrics_dict:
            metrics[metric_name] = metrics_dict[metric_name](self.decision_function(X),Y)
        return metrics

    def fit(self, X, y, num_epochs=10, batch_size=100, compute_metrics=False, metrics_dict=None):
        if compute_metrics:
            metrics_history = {name: [] for name in metrics_dict}
            metrics = self.compute_metrics(X, y, metrics_dict)
            for name in metrics_dict:
                metrics_history[name].append(metrics[name])
        else:
            metrics_history = None

        for _ in range(num_epochs):
            indices = np.random.permutation(len(X))
            batches = np.array_split(indices, np.ceil(len(X) / batch_size))

            for idx in batches:
                grads = self.loss_grad(X[idx], y[idx])
                self.optimizer.update(self._get_params(), grads)

                if compute_metrics:
                    metrics = self.compute_metrics(X, y, metrics_dict)
                    for name in metrics_dict:
                        metrics_history[name].append(metrics[name])

        return metrics_history
    
