"""Sigmoid activation: squashes real values into (0, 1)."""

import numpy as np

from nn.module import Module


class Sigmoid(Module):
    """Sigmoid activation, applied elementwise: 1 / (1 + e^{-x})."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute sigmoid elementwise, and remember the output.

        Args:
            x (np.ndarray): input, any shape.

        Returns:
            np.ndarray: sigmoid(x), elementwise, same shape as x.
        """
        positive = x >= 0
        z = np.exp(-np.abs(x))
        self.a = np.where(positive, 1 / (1 + z), z / (1 + z))
        return self.a

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradients given the upstream gradient.

        Args:
            grad_output (np.ndarray): gradient of the loss with
                respect to this layer's output, same shape as
                the original input to forward.

        Returns:
            np.ndarray: gradient of the loss with respect to
                this layer's input, same shape as grad_output.
        """
        return grad_output * self.a * (1 - self.a)
