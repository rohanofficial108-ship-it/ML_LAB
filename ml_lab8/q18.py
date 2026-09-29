# =========================================================
# A1: Activation Functions & Utility Modules
# =========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def summation_unit(inputs, weights, bias):
    return np.dot(inputs, weights) + bias

def step_function(z):
    return 1 if z >= 0 else 0

def bipolar_step_function(z):
    return 1 if z > 0 else (-1 if z < 0 else 0)

def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))

def tanh_function(z):
    return np.tanh(z)

def relu_function(z):
    return np.maximum(0, z)

def leaky_relu_function(z, alpha=0.01):
    return np.where(z > 0, z, z * alpha)

def calculate_error(target, output):
    return 0.5 * (target - output) ** 2

# Helper (used by A2-A5)
def train_perceptron(X, y, weights, lr, activation_func,
                     error_threshold=0.002, max_ep=1000):
    epochs = 0
    errors = []
    while epochs < max_ep:
        total_error = 0
        for i in range(len(X)):
            inputs = np.insert(X[i], 0, 1)
            z = summation_unit(inputs, weights[1:], weights[0])
            output = activation_func(z)
            error = y[i] - output
            total_error += error ** 2
            weights[1:] += lr * error * X[i]
            weights[0]  += lr * error
        errors.append(total_error)
        if total_error <= error_threshold:
            break
        epochs += 1
    return weights, epochs, errors

# =========================================================
# A2: AND Gate with Step Activation
# =========================================================
data = np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,1]])
X, y = data[:, :2], data[:, 2]
W = np.array([0, 0.2, -0.75])
alpha = 0.05

final_weights, epochs_taken, error_history = train_perceptron(
    X, y, W.copy(), alpha, step_function)
print(f"A2: AND converged in {epochs_taken} epochs")

plt.figure()
plt.plot(error_history)
plt.title("A2: AND Gate (Step)")
plt.xlabel("Epoch"); plt.ylabel("Error")
plt.show()

# =========================================================
# A3: AND Gate with Bi-Polar, Sigmoid, ReLU
# =========================================================
results = {}
for name, fn in [("Bi-Polar", bipolar_step_function),
                 ("Sigmoid",  sigmoid_function),
                 ("ReLU",     relu_function)]:
    _, ep, err = train_perceptron(X, y, W.copy(), alpha, fn)
    results[name] = (ep, err)
    print(f"A3: AND {name} -> {ep} epochs")

plt.figure()
for name, (ep, err) in results.items():
    plt.plot(err, label=name)
plt.legend(); plt.title("A3: AND Gate Activation Comparison")
plt.xlabel("Epoch"); plt.ylabel("Error")
plt.show()

# =========================================================
# A4: Varying Learning Rate (AND)
# =========================================================
lrs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
iters = [train_perceptron(X, y, W.copy(), lr, step_function)[1] for lr in lrs]
plt.figure()
plt.plot(lrs, iters, marker='o')
plt.title("A4: Learning Rate vs Iterations")
plt.xlabel("Learning Rate"); plt.ylabel("Iterations")
plt.show()

# =========================================================
# A5: XOR Gate
# =========================================================
data_xor = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
X_xor, y_xor = data_xor[:, :2], data_xor[:, 2]

_, ep_step, _ = train_perceptron(X_xor, y_xor, W.copy(), alpha, step_function)
_, ep_sig,  _ = train_perceptron(X_xor, y_xor, W.copy(), alpha, sigmoid_function)
print(f"A5: XOR Step {ep_step} epochs | XOR Sigmoid {ep_sig} epochs (won't converge)")

# =========================================================
# A6 + A7: Iris Perceptron vs Pseudo-Inverse
# =========================================================
df = pd.read_csv('Iris.csv')
df['Target'] = df['Species'].apply(lambda x: 1 if x == 'Iris-setosa' else 0)

features = df[['SepalLengthCm','SepalWidthCm']].values
target   = df['Target'].values
features = (features - features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0))

def train_perceptron_batch(X, y, weights, lr, activation_func,
                           error_threshold=0.002, max_ep=1000):
    epochs, errors = 0, []
    while epochs < max_ep:
        total_error = 0
        for i in range(len(X)):
            inputs = np.insert(X[i], 0, 1)
            z = summation_unit(inputs, weights[1:], weights[0])
            output = activation_func(z)
            error = y[i] - output
            total_error += error ** 2
            weights[1:] += lr * error * X[i]
            weights[0]  += lr * error
        errors.append(total_error)
        if total_error <= error_threshold:
            break
        epochs += 1
    return weights, epochs, errors

W_iris = np.random.rand(3) * 0.1
w_iris_final, ep_iris, _ = train_perceptron_batch(features, target, W_iris, 0.05, sigmoid_function)
print(f"A6: Iris Perceptron converged in {ep_iris} epochs")

X_bias = np.hstack([np.ones((features.shape[0],1)), features])
weights_pinv = np.linalg.pinv(X_bias) @ target
print(f"A7: Pseudo-Inverse weights = {weights_pinv}")

# =========================================================
# A8 + A9 + A10: Backprop
# =========================================================
def train_backprop(X, y, lr=0.05, max_ep=1000, err_thresh=0.002):
    np.random.seed(42)
    V  = np.random.uniform(-0.5, 0.5, (2,2))
    V0 = np.random.uniform(-0.5, 0.5, 2)
    W_  = np.random.uniform(-0.5, 0.5, 2)
    W0_ = np.random.uniform(-0.5, 0.5, 1)
    errors = []
    for epoch in range(max_ep):
        total_error = 0
        for i in range(len(X)):
            h_in  = np.dot(X[i], V) + V0
            h_out = sigmoid_function(h_in)
            o_in  = np.dot(h_out, W_) + W0_
            o_out = sigmoid_function(o_in)
            error = y[i] - o_out
            total_error += error ** 2

            delta_o = o_out * (1 - o_out) * error
            delta_h = h_out * (1 - h_out) * (delta_o * W_)

            W_  += lr * delta_o * h_out
            W0_ += lr * delta_o
            V   += lr * np.outer(X[i], delta_h)
            V0  += lr * delta_h
        errors.append(total_error)
        if total_error <= err_thresh:
            break
    return epoch, errors

ep_bp_and, _ = train_backprop(X, y)
ep_bp_xor, err_bp_xor = train_backprop(X_xor, y_xor)
print(f"A8: Backprop AND -> {ep_bp_and} epochs")
print(f"A9: Backprop XOR -> {ep_bp_xor} epochs")

plt.figure(); plt.plot(err_bp_xor); plt.title("A9: Backprop XOR Error"); plt.show()

def train_backprop_multi(X, y_multi, lr=0.05, max_ep=1000, err_thresh=0.002):
    np.random.seed(42)
    V  = np.random.uniform(-0.5, 0.5, (2,2))
    V0 = np.random.uniform(-0.5, 0.5, 2)
    W_  = np.random.uniform(-0.5, 0.5, (2,2))
    W0_ = np.random.uniform(-0.5, 0.5, 2)
    for epoch in range(max_ep):
        total_error = 0
        for i in range(len(X)):
            h_in  = np.dot(X[i], V) + V0
            h_out = sigmoid_function(h_in)
            o_in  = np.dot(h_out, W_) + W0_
            o_out = sigmoid_function(o_in)
            error = y_multi[i] - o_out
            total_error += np.sum(error ** 2)

            delta_o = o_out * (1 - o_out) * error
            delta_h = h_out * (1 - h_out) * np.dot(delta_o, W_.T)

            W_  += lr * np.outer(h_out, delta_o)
            W0_ += lr * delta_o
            V   += lr * np.outer(X[i], delta_h)
            V0  += lr * delta_h
        if total_error <= err_thresh:
            break
    return epoch

y_and_multi = np.array([[1,0],[1,0],[1,0],[0,1]])
y_xor_multi = np.array([[1,0],[0,1],[0,1],[1,0]])
print(f"A10: Multi-output AND -> {train_backprop_multi(X, y_and_multi)} epochs")
print(f"A10: Multi-output XOR -> {train_backprop_multi(X_xor, y_xor_multi)} epochs")

# =========================================================
# A11 + A12: Sklearn MLP
# =========================================================
clf_and = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic',
                        solver='lbfgs', max_iter=1000, random_state=1)
clf_and.fit(X, y)
print(f"A11: Sklearn AND accuracy = {clf_and.score(X, y)*100:.2f}%")

clf_xor = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic',
                        solver='lbfgs', max_iter=1000, random_state=1)
clf_xor.fit(X_xor, y_xor)
print(f"A11: Sklearn XOR accuracy = {clf_xor.score(X_xor, y_xor)*100:.2f}%")

X_iris_full = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].values
y_iris_sp   = df['Species'].values

Xtr, Xte, ytr, yte = train_test_split(X_iris_full, y_iris_sp,
                                      test_size=0.3, random_state=42)
sc = StandardScaler()
Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)

mlp = MLPClassifier(hidden_layer_sizes=(10,10), activation='relu',
                    solver='adam', max_iter=1000, random_state=42)
mlp.fit(Xtr, ytr)
print(f"A12: Iris MLP accuracy = {accuracy_score(yte, mlp.predict(Xte))*100:.2f}%")