"""
neural_network.py
A simple feedforward neural network that will serve as each agent's brain.
Weights will be evolved via genetic mutation in a future milestone.
For now this establishes the architecture and weight interface.

Architecture:
    Input layer  -> Hidden layer -> Output layer
    (8 inputs)      (12 neurons)    (4 outputs)

Planned Inputs (8):
    0: distance to nearest enemy (normalized)
    1: angle to nearest enemy (normalized)
    2: own health (normalized)
    3: enemy health (normalized)
    4: distance to nearest wall (normalized)
    5: velocity x (normalized)
    6: velocity y (normalized)
    7: bias (always 1.0)

Planned Outputs (4):
    0: move x  (-1 to 1)
    1: move y  (-1 to 1)
    2: attack  (> 0.5 triggers attack)
    3: flee    (> 0.5 activates flee mode)
"""

import numpy as np


class NeuralNetwork:
    INPUT_SIZE  = 8
    HIDDEN_SIZE = 12
    OUTPUT_SIZE = 4

    def __init__(self, weights=None):
        if weights is not None:
            self.set_weights(weights)
        else:
            self.randomize()

    def randomize(self):
        """Initialize weights using Xavier initialization."""
        self.W1 = np.random.randn(self.INPUT_SIZE, self.HIDDEN_SIZE) * np.sqrt(2.0 / self.INPUT_SIZE)
        self.b1 = np.zeros(self.HIDDEN_SIZE)
        self.W2 = np.random.randn(self.HIDDEN_SIZE, self.OUTPUT_SIZE) * np.sqrt(2.0 / self.HIDDEN_SIZE)
        self.b2 = np.zeros(self.OUTPUT_SIZE)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Run a forward pass and return output activations."""
        x = np.tanh(inputs @ self.W1 + self.b1)
        x = np.tanh(x @ self.W2 + self.b2)
        return x

    def get_weights(self) -> np.ndarray:
        """Flatten all weights into a 1D array for future genetic operations."""
        return np.concatenate([
            self.W1.flatten(),
            self.b1.flatten(),
            self.W2.flatten(),
            self.b2.flatten()
        ])

    def set_weights(self, weights: np.ndarray):
        """Restore network from a flat 1D weight array."""
        idx = 0
        size = self.INPUT_SIZE * self.HIDDEN_SIZE
        self.W1 = weights[idx:idx + size].reshape(self.INPUT_SIZE, self.HIDDEN_SIZE); idx += size
        self.b1 = weights[idx:idx + self.HIDDEN_SIZE]; idx += self.HIDDEN_SIZE
        size = self.HIDDEN_SIZE * self.OUTPUT_SIZE
        self.W2 = weights[idx:idx + size].reshape(self.HIDDEN_SIZE, self.OUTPUT_SIZE); idx += size
        self.b2 = weights[idx:idx + self.OUTPUT_SIZE]

    def copy(self) -> "NeuralNetwork":
        return NeuralNetwork(weights=self.get_weights().copy())

    @property
    def weight_count(self) -> int:
        return (
            self.INPUT_SIZE * self.HIDDEN_SIZE + self.HIDDEN_SIZE +
            self.HIDDEN_SIZE * self.OUTPUT_SIZE + self.OUTPUT_SIZE
        )
