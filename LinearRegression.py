import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# PART 1: Generate simple linear data with noise
# ----------------------------------------------------------

np.random.seed(5) # Default Randomseed for now (change the number if you want different results)
# We need to seed it because computers have no true random

# Create 80 evenly spaced x-values from -2, 2
x = np.linspace(-2, 2, 80)

# Underlying relationship (This is what we WANT, the model doesnt know this it will try to approx. it)
true_w = 2.5
true_b = -1.0

# Create y-values using the line, but add random noise
y = true_w * x + true_b + 0.6 * np.random.randn(len(x))

# ----------------------------------------------------------
# PART 2: Define the model we want to train
# Model: y_pred = w*x + b
# ----------------------------------------------------------

def predict(w, b, x):
    """
    Computes predicted y-values for given w, b, and x.
    This is the 'model' function
    """
    return w * x + b

# ----------------------------------------------------------
# PART 3: Define the loss function (Mean Squared Error) (This is the typical loss calculator)
# ----------------------------------------------------------

def mse_loss(w, b, x, y):
    """
    Computes Mean Squared Error (MSE)
    MSE = average of (pred - true)^2
    """
    predictions = predict(w, b, x)
    errors = predictions - y
    return np.mean(errors ** 2) / 2

# ----------------------------------------------------------
# PART 4: Compute the gradient (slope) of the loss function
# with respect to w and b
# ----------------------------------------------------------

def compute_gradients(w, b, x, y):
    """
    Computes the partial derivatives of MSE with respect to w and b.
    These tell us how to adjust w and b to reduce the loss.
    
    *If youre curious about this take calc3 (its hard)*

    dL/dw = average((predicted - actual) * x)
    dL/db = average(predicted - actual)
    """
    predictions = predict(w, b, x)
    errors = predictions - y

    dw = np.mean(errors * x)
    db = np.mean(errors)

    return dw, db

# ----------------------------------------------------------
# PART 5: Train using Gradient Descent
# ----------------------------------------------------------

# Start with intentionally bad guesses
w = -3.0
b = 3.0

learning_rate = 0.15 # How big the correction steps should be
steps = 60 # amount of attempts of the model (epochs)

# Just to store the history for plotting later
history_w = []
history_b = []
history_loss = []

for step in range(steps):
    # compute loss
    loss = mse_loss(w, b, x, y)

    # save values
    history_w.append(w)
    history_b.append(b)
    history_loss.append(loss)

    # compute gradient (direction of steepest increase)
    dw, db = compute_gradients(w, b, x, y)

    # Change parameters to the opposite direction of the gradient
    w -= learning_rate * dw
    b -= learning_rate * db

# ----------------------------------------------------------
# PART 6: Plot the results
# ----------------------------------------------------------

# === Plot 1: Data + learned best-fit line ===
plt.figure()

plt.subplot(1,2,1)
plt.scatter(x, y, s=20, label="Data Points")
plt.plot(x, true_w*x + true_b, label="True Line", linewidth=2)
plt.plot(x, predict(w,b,x), label="Learned Line", linewidth=2)
plt.title("Linear Regression Fit Over Data")
plt.legend()
plt.grid(True)

# === Plot 2: Loss vs training step ===
plt.subplot(1,2,2)
plt.plot(history_loss)
plt.title("Loss Over Training Steps")
plt.xlabel("Step")
plt.ylabel("Loss (MSE)")
plt.grid(True)

plt.show()

# ----------------------------------------------------------
# PART 7: Things to test!
# ----------------------------------------------------------

"""
Here are some cool ideas you can use to try and make this your own! Please have at least one of these modifications by monday please :)!

DECAYED LEARNING
“Implement a learning rate schedule where the learning rate decays over time (for example, lr = lr0 / (1 + k * step)).
Train with constant lr vs decaying lr and compare their loss curves.”

QUADRATICS:
“Modify the model so the prediction is:
y_hat = w2 * x² + w1 * x + b
Train with gradient descent and plot the curve you learn.
Does the model fit noisy data better?”

OUTLIERS
“Add 3-5 large outliers to the dataset and retrain your model.
Plot the fitted line before and after adding these outliers.
Explain why mean squared error is very sensitive to outliers.”

BONUS POINTS FOR THIS ONE
“Modify the code to generate a 3D plot of the loss surface over a grid of w and b values. Then overlay the gradient descent path you computed.
Describe how the shape of the surface affects the movement of the parameters.”
"""
