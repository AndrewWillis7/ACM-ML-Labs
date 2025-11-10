# In this lab youre gonna build a really basic nueral net to train on spiral data

# PREREQUISITES (LinearRegression.py)

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0) # Any seed you want

# ----------------------------------------------------------
# PART 1: Create a challenging dataset (two spirals)
# ----------------------------------------------------------

# I dont really care if you understand the spiral math, this doesnt pertain to the nn code (this is just the data)
def make_spirals(n_points=200, noise=0.2):
    """
    Creates two spiral-shaped clusters for binary classification.
    Good for testing nonlinear models.
    """
    n = n_points
    theta = np.sqrt(np.random.rand(n)) * 4 * np.pi
    r = theta

    # first spiral
    x1 = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    x1 += noise * np.random.randn(n, 2)

    # second spiral
    x2 = np.stack([-r * np.cos(theta), -r * np.sin(theta)], axis=1)
    x2 += noise * np.random.randn(n, 2)

    X = np.vstack([x1, x2])
    y = np.hstack([np.zeros(n), np.ones(n)]).astype(int)

    # Normalize data for training stability
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    return X, y

def ideal_spirals(n_points=400):
    """
    Generates noise-free versions of the spirals.
    These represent the underlying structure we want students to visualize.
    """
    n = n_points
    theta = np.linspace(0, 4*np.pi, n)
    r = theta

    # Ideal (noise-free) spiral 1
    s1 = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

    # Ideal spiral 2 (opposite direction)
    s2 = np.stack([-r * np.cos(theta), -r * np.sin(theta)], axis=1)

    # Normalize to match the dataset scaling
    all_points = np.vstack([s1, s2])
    mean = all_points.mean(axis=0)
    std = all_points.std(axis=0)

    s1 = (s1 - mean) / (std + 1e-8)
    s2 = (s2 - mean) / (std + 1e-8)

    return s1, s2

# Make the spirals
X, y = make_spirals()

# Convert labels to one-hot (needed for softmax loss)
Y_onehot = np.eye(2)[y]

# ----------------------------------------------------------
# PART 2: Define a SIMPLE Neural Network (2 → 16 → 2)
# ----------------------------------------------------------

def initialize_parameters():
    """
    Randomly Initialize all weights and biases
    """
    W1 = np.random.randn(2, 16) * 0.5 # Layer 1 weights
    b1 = np.zeros((1, 16)) # Layer 1 bias
    W2 = np.random.randn(16, 2) * 0.5 # Layer 2 weights
    b2 = np.zeros((1, 2)) # Layer 2 bias
    return W1, b1, W2, b2

# Activation function (non-linear)
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def softmax(z):
    """
    Converts raw scores into probabilities.
    """
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def forward_pass(X, W1, b1, W2, b2):
    """
    Computes the forward pass of the network:
        X -> hidden layer -> output layer -> probabilities
    """
    z1 = X @ W1 + b1
    h = tanh(z1)
    logits = h @ W2 + b2
    probs = softmax(logits)
    return z1, h, logits, probs

def cross_entropy(probs, Y):
    """
    Cross-entropy loss for classification.
    """
    eps = 1e-12
    return -np.mean(np.sum(Y * np.log(probs + eps), axis=1))

def backwards_pass(X, Y, z1, h, probs, W1, W2):
    """
    Computes gradients using backpropagation.
    This tells us how each weight should change.
    """
    N = len(X)

    # gradient at output layer
    dlogits = (probs - Y) / N

    # Gradients for W2, b2
    dW2 = h.T @ dlogits
    db2 = np.sum(dlogits, axis=0, keepdims=True)

    # Backpropogate into hidden layer
    dh = dlogits @ W2.T
    dz1 = dh * tanh_derivative(z1)

    # Gradients for W1, b1
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2

# ----------------------------------------------------------
# PART 3: Train the network
# ----------------------------------------------------------

W1, b1, W2, b2 = initialize_parameters()

learning_rate = 0.1
epochs = 200

loss_history = []

for epoch in range(epochs):
    # Step 1: forward pass
    z1, h, logits, probs = forward_pass(X, W1, b1, W2, b2)

    # Step 2: Compute Loss
    loss = cross_entropy(probs, Y_onehot)
    loss_history.append(loss)

    # Step 3: Backwards pass (compute Gradients)
    dW1, db1, dW2, db2 = backwards_pass(X, Y_onehot, z1, h, probs, W1, W2)

    # Step 4: Gradient Descent update
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

# ----------------------------------------------------------
# PART 4: Plot loss and decision boundary
# ----------------------------------------------------------

# Plot loss curve
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.grid(True)

# Decision boundary visualization
def plot_boundary():
    # Create a grid covering the data
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Forward pass on the grid
    _, _, _, grid_probs = forward_pass(grid_points, W1, b1, W2, b2)
    
    # Class 1 probability
    Z = grid_probs[:,1].reshape(xx.shape)
    
    # Plot
    plt.subplot(1,2,2)
    plt.contourf(xx, yy, Z, levels=50, cmap='coolwarm', alpha=0.8)
    plt.scatter(X[:,0], X[:,1], c=y, s=10, cmap='coolwarm', edgecolors='k')
    plt.title("Learned Decision Boundary")

plot_boundary()

# Generate ideal spirals
s1, s2 = ideal_spirals()

# Plot them for reference
plt.figure(figsize=(5,5))
plt.plot(s1[:,0], s1[:,1], 'b', linewidth=2, label="Ideal Spiral 0")
plt.plot(s2[:,0], s2[:,1], 'r', linewidth=2, label="Ideal Spiral 1")
plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm', s=10, alpha=0.4)
plt.legend()
plt.title("Ideal Spirals vs. Noisy Spiral Dataset")
plt.show()