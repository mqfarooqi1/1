"""
=============================================================================
 HANDWRITTEN DIGIT RECOGNITION  —  a Neural Network built FROM SCRATCH
=============================================================================

No TensorFlow, no PyTorch, no Keras. Just NumPy and math.

This program trains a real neural network to recognise handwritten digits
(0-9). Everything that a deep-learning library normally hides is written out
by hand here so you can SEE how a network actually learns:

   * forward pass      (how the network makes a prediction)
   * cross-entropy loss (how wrong the prediction is)
   * backpropagation    (how the error flows backward to each weight)
   * gradient descent   (how the weights are nudged to improve)

The data is the classic 8x8 "digits" dataset that ships INSIDE scikit-learn,
so nothing is downloaded and it runs in a couple of seconds.

NETWORK ARCHITECTURE
--------------------
        64 inputs            64 hidden neurons          10 outputs
   (8x8 pixel image)  -->   (ReLU activation)    -->   (one per digit,
                                                         softmax = probabilities)

SETUP (run once)
----------------
    pip install numpy scikit-learn

RUN
---
    python my.py
=============================================================================
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# Make results reproducible (same "random" numbers every run).
rng = np.random.default_rng(42)


# --------------------------------------------------------------------------
# Math building blocks
# --------------------------------------------------------------------------
def relu(z):
    """ReLU activation: keep positives, zero-out negatives."""
    return np.maximum(0, z)


def relu_derivative(z):
    """Gradient of ReLU: 1 where input was positive, else 0."""
    return (z > 0).astype(float)


def softmax(z):
    """Turn raw scores into probabilities that sum to 1 (per row)."""
    z = z - z.max(axis=1, keepdims=True)        # subtract max for stability
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def one_hot(labels, num_classes):
    """Turn label 3 into [0,0,0,1,0,0,0,0,0,0]."""
    encoded = np.zeros((labels.size, num_classes))
    encoded[np.arange(labels.size), labels] = 1
    return encoded


# --------------------------------------------------------------------------
# The neural network
# --------------------------------------------------------------------------
class NeuralNetwork:
    def __init__(self, n_inputs: int, n_hidden: int, n_outputs: int):
        # He initialisation keeps signal from vanishing/exploding through ReLU.
        self.W1 = rng.standard_normal((n_inputs, n_hidden)) * np.sqrt(2 / n_inputs)
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = rng.standard_normal((n_hidden, n_outputs)) * np.sqrt(2 / n_hidden)
        self.b2 = np.zeros((1, n_outputs))

    # ---- FORWARD PASS: input -> prediction -------------------------------
    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1          # linear layer 1
        self.a1 = relu(self.z1)                   # activation
        self.z2 = self.a1 @ self.W2 + self.b2     # linear layer 2
        self.a2 = softmax(self.z2)                # probabilities
        return self.a2

    # ---- BACKWARD PASS: how should every weight change? ------------------
    def backward(self, X, Y, learning_rate: float):
        n = X.shape[0]

        # Gradient of (softmax + cross-entropy) is simply (prediction - truth).
        dz2 = (self.a2 - Y) / n
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)

        # Propagate the error back through the hidden layer (chain rule).
        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_derivative(self.z1)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)

        # GRADIENT DESCENT: step every weight a little against its gradient.
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def predict(self, X):
        """Return the most likely digit for each image."""
        return self.forward(X).argmax(axis=1)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def cross_entropy(probs, Y):
    """Average penalty for being confident and wrong."""
    return -np.mean(np.sum(Y * np.log(probs + 1e-9), axis=1))


def accuracy(predictions, labels):
    return np.mean(predictions == labels)


def ascii_art(image_8x8):
    """Render an 8x8 digit as text so we can 'see' it in the terminal."""
    shades = " .:-=+*#%@"
    img = image_8x8.reshape(8, 8)
    lines = []
    for row in img:
        lines.append("".join(shades[min(int(p / 16 * 9), 9)] for p in row))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Train and evaluate
# --------------------------------------------------------------------------
def main() -> None:
    # 1) Load real handwritten-digit images (1,797 of them, 8x8 pixels each).
    digits = load_digits()
    X = digits.data / 16.0          # scale pixel values from 0-16 to 0-1
    y = digits.target              # the correct digit for each image (0-9)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    Y_train = one_hot(y_train, 10)

    print(f"Training images: {X_train.shape[0]}   Test images: {X_test.shape[0]}")
    print("Training a neural network from scratch...\n")

    # 2) Build the network and train it with mini-batch gradient descent.
    net = NeuralNetwork(n_inputs=64, n_hidden=64, n_outputs=10)
    epochs = 60
    batch_size = 32
    learning_rate = 0.5

    for epoch in range(1, epochs + 1):
        # Shuffle the training data each epoch.
        order = rng.permutation(X_train.shape[0])
        X_shuf, Y_shuf = X_train[order], Y_train[order]

        # Walk through the data in small batches.
        for start in range(0, X_shuf.shape[0], batch_size):
            xb = X_shuf[start:start + batch_size]
            yb = Y_shuf[start:start + batch_size]
            net.forward(xb)
            net.backward(xb, yb, learning_rate)

        # Report progress every 10 epochs.
        if epoch % 10 == 0 or epoch == 1:
            train_loss = cross_entropy(net.forward(X_train), Y_train)
            train_acc = accuracy(net.predict(X_train), y_train)
            print(f"  epoch {epoch:>3}/{epochs} | "
                  f"loss {train_loss:.3f} | train accuracy {train_acc:.1%}")

    # 3) Final score on images the network has NEVER seen.
    test_acc = accuracy(net.predict(X_test), y_test)
    print(f"\nFINAL TEST ACCURACY: {test_acc:.1%}  "
          f"(on {X_test.shape[0]} unseen images)")

    # 4) Show a few predictions so you can see it in action.
    print("\nSample predictions on unseen digits:")
    print("=" * 40)
    for i in range(5):
        guess = net.predict(X_test[i:i + 1])[0]
        truth = y_test[i]
        mark = "OK " if guess == truth else "X  "
        print(f"\n{mark} predicted: {guess}   actual: {truth}")
        print(ascii_art(X_test[i] * 16))


if __name__ == "__main__":
    main()
