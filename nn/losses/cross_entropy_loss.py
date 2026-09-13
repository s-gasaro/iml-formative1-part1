"""Binary cross-entropy loss."""

import numpy as np


class CrossEntropyLoss:
    """Binary cross-entropy loss for a single output probability.

    Does not subclass Module -- its forward/backward signatures
    don't match the shared layer/activation contract, and it has
    no learnable parameters.
    """

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute the average binary cross-entropy loss.

        Args:
            predictions (np.ndarray): predicted probabilities,
                shape (m,) or (m, 1).
            targets (np.ndarray): true labels, same shape as
                predictions, values 0 or 1.

        Returns:
            float: the scalar loss, averaged over the batch.
        """
        self.predictions = np.clip(predictions, 1e-12, 1 - 1e-12)
        self.targets = targets
        m = targets.shape[0]
        loss = -(targets * np.log(self.predictions) +
                 (1 - targets) * np.log(1 - self.predictions))
        return float(np.sum(loss) / m)

    def backward(self) -> np.ndarray:
        """Compute the gradient of the loss w.r.t. predictions.

        Returns:
            np.ndarray: dL/da, same shape as the predictions
                passed to forward.
        """
        m = self.targets.shape[0]
        return -(self.targets / self.predictions -
                  (1 - self.targets) / (1 - self.predictions)) / m
