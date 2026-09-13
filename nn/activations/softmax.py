"""Softmax activation: converts logits into a probability distribution."""

import numpy as np

from nn.module import Module


class Softmax(Module):
    """Softmax activation, applied row-wise to a batch of logits.

    Unlike ReLU or Sigmoid, each output depends on every logit in
    its own row, not just the matching input.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities for a batch of logits.

        Args:
            x (np.ndarray): logits, shape (batch_size, C).

        Returns:
            np.ndarray: probabilities, shape (batch_size, C).
                Each row sums to 1.
        """
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.a = exp / np.sum(exp, axis=1, keepdims=True)
        return self.a

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradients given the upstream gradient.

        Args:
            grad_output (np.ndarray): gradient of the loss with
                respect to this layer's output, shape (batch_size, C).

        Returns:
            np.ndarray: gradient of the loss with respect to this
                layer's input (the logits), shape (batch_size, C).
        """
        dot = np.sum(grad_output * self.a, axis=1, keepdims=True)
        return self.a * (grad_output - dot)
