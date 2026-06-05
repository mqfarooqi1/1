# Handwritten Digit Recognition — Neural Network from Scratch 🧠

A 2-layer neural network that recognizes handwritten digits (0–9), built in
**pure NumPy** — no TensorFlow, no PyTorch, no Keras. Every core deep-learning
mechanism is hand-coded so you can see exactly how a network learns.

Trains in ~2 seconds on CPU and reaches **~97% accuracy** on unseen images.

## What's implemented by hand

| Concept | Where |
|---|---|
| **Forward pass** (linear → ReLU → linear → softmax) | `NeuralNetwork.forward` |
| **Cross-entropy loss** | `cross_entropy` |
| **Backpropagation** (chain rule, by hand) | `NeuralNetwork.backward` |
| **Mini-batch gradient descent** | `main` training loop |
| **He weight initialization** | `NeuralNetwork.__init__` |

## Architecture

```
   64 inputs            64 hidden neurons          10 outputs
(8x8 pixel image)  -->   (ReLU activation)   -->   (one per digit,
                                                     softmax = probabilities)
```

The dataset is scikit-learn's bundled 8×8 `digits` dataset (1,797 images), so
**nothing is downloaded** and the project runs offline out of the box.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python my.py
```

## Example output

```
Training images: 1437   Test images: 360
Training a neural network from scratch...

  epoch   1/60 | loss 0.339 | train accuracy 90.3%
  epoch  10/60 | loss 0.047 | train accuracy 98.7%
  ...
  epoch  60/60 | loss 0.003 | train accuracy 100.0%

FINAL TEST ACCURACY: 97.5%  (on 360 unseen images)

Sample predictions on unseen digits:
========================================

OK  predicted: 5   actual: 5
  =*#*.
  *: .
  @.:
 :@#*%:
  :  ==
     *:
 -# .%
  =#@:
```

Each prediction is shown alongside an ASCII-art render of the digit, so you can
*see* what the network is classifying.

## Tweak it

All hyperparameters live at the top of the `main()` function in
[`my.py`](my.py):

- `epochs` — how many passes over the training data
- `batch_size` — mini-batch size
- `learning_rate` — gradient-descent step size
- `n_hidden` (in `NeuralNetwork(...)`) — width of the hidden layer

## Requirements

- Python 3.9+
- numpy
- scikit-learn

---

Built by [Muhammad Farooqi](https://github.com/mqfarooqi1).
