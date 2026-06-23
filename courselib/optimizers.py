import numpy as np

class Optimizer:
    """
    Base optimizer class.
    """

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        """
        Update parameters based on gradients.
        This method should be overridden by subclasses.

        Parameters:
        - params: list or dict of parameters (e.g., weights)
        - grads: list or dict of gradients (same structure as params)
        """
        raise NotImplementedError("`update` must be implemented by the subclass.")


class GDOptimizer(Optimizer):
    """
    Gradient descent optimizer with optional learning rate schedule.

    Parameters:
    - learning_rate (float): Initial learning rate
    - schedule_fn (callable): Function(step) → new_learning_rate
    """

    def __init__(self, learning_rate=0.01, schedule_fn=None):
        super().__init__(learning_rate)
        self.schedule_fn = schedule_fn
        self.step = 0

    def update(self, params, grads):
        if self.schedule_fn is not None:
            self.step += 1
            self.learning_rate = self.schedule_fn(self.step)

        for key in params:
            np.subtract(params[key], self.learning_rate * grads[key], out=params[key])

