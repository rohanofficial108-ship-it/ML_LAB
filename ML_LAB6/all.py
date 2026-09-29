import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- A1: Modular Design (AI Assisted) ---
# Prompt used: "Generate a modular Python class for KNN using numpy vectorization, including fit, predict, score, and weighted voting."

class GenAI_KNN:
    def __init__(self, k=3, weighted=False, metric='euclidean'):
        self.k = k
        self.weighted = weighted
        self.metric = metric
        
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
    def _calculate_distance(self, X_test):
        # Vectorized Euclidean distance calculation
        # X_test: (M, D), X_train: (N, D) -> Output: (M, N)
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((X_test[:, np.newaxis] - self.X_train)**2, axis=2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(X_test[:, np.newaxis] - self.X_train), axis=2)

    def predict(self, X_test):
        X_test = np.array(X_test)
        
        # A1c. Distance Calculation
        distances = self._calculate_distance(X_test)
        
        # A1d. Sorting (Using numpy argsort as 'AI' optimized sorting)
        # Get indices of k nearest neighbors
        nearest_indices = np.argsort(distances, axis=1)[:, :self.k]
        
        predictions = []
        
        for i, indices in enumerate(nearest_indices):
            labels = self.y_train[indices]
            
            # A1f. Class Evaluation
            if self.weighted:
                # A2. Weighted kNN
                dists = distances[i, indices]
                # Avoid division by zero
                weights = 1 / (dists + 1e-5)
                # Sum weights per class
                unique_labels = np.unique(labels)
                class_weights = {lbl: np.sum(weights[labels == lbl]) for lbl in unique_labels}
                pred = max(class_weights, key=class_weights.get)
            else:
                # Majority Voting
                # np.unique returns sorted unique values, so tie-breaking is consistent
                values, counts = np.unique(labels, return_counts=True)
                pred = values[np.argmax(counts)]
            
            predictions.append(pred)
            
        return np.array(predictions)

    def score(self, X_test, y_test):
        preds = self.predict(X_test)
        return np.mean(preds == np.array(y_test))

# --- Data Loading ---
# AI Prompt: "Load Iris.csv, drop Id, encode target, and filter for binary classification."

df = pd.read_csv('Iris.csv')
df = df.drop('Id', axis=1)

# Filter for two classes (Setosa and Versicolor)
df = df[df['Species'].isin(['Iris-setosa', 'Iris-versicolor'])]

# Encode Labels
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# A3. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Comparison ---

# 1. My GenAI Implementation
genai_model = GenAI_KNN(k=3)
start = time.time()
genai_model.fit(X_train, y_train)
genai_preds = genai_model.predict(X_test)
genai_time = time.time() - start

# 2. Sklearn Implementation
sk_model = KNeighborsClassifier(n_neighbors=3)
start = time.time()
sk_model.fit(X_train, y_train)
sk_preds = sk_model.predict(X_test)
sk_time = time.time() - start

# 3. Manual Implementation (Referencing Version 1 logic conceptually, or re-instantiate here)
# For the sake of this script, we simulate "Manual" by forcing a non-vectorized loop if needed,
# but here we will just compare GenAI vs Sklearn vs Weighted.

# Weighted GenAI
genai_weighted = GenAI_KNN(k=3, weighted=True)
genai_weighted.fit(X_train, y_train)
weighted_preds = genai_weighted.predict(X_test)

# --- A8/A9: Metrics and Plotting ---

def get_metrics(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='binary'),
        'Recall': recall_score(y_true, y_pred, average='binary'),
        'F1': f1_score(y_true, y_pred, average='binary')
    }

metrics_genai = get_metrics(y_test, genai_preds)
metrics_sk = get_metrics(y_test, sk_preds)
metrics_weighted = get_metrics(y_test, weighted_preds)

# Timing 10 runs (A3 requirement)
def time_model(model, X, y, X_t, is_sklearn=False):
    start = time.time()
    if is_sklearn:
        model.fit(X, y)
        model.predict(X_t)
    else:
        model.fit(X, y)
        model.predict(X_t)
    return time.time() - start

t_genai = []
t_sk = []
t_weighted = []

for _ in range(10):
    t_genai.append(time_model(GenAI_KNN(k=3), X_train, y_train, X_test))
    t_sk.append(time_model(KNeighborsClassifier(n_neighbors=3), X_train, y_train, X_test, True))
    t_weighted.append(time_model(GenAI_KNN(k=3, weighted=True), X_train, y_train, X_test))

avg_t_genai = np.mean(t_genai)
avg_t_sk = np.mean(t_sk)
avg_t_weighted = np.mean(t_weighted)

# Table Output
print("\n--- Performance Comparison Table ---")
print(f"{'Model':<20} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'Time (s)':<10}")
print("-" * 75)
print(f"{'GenAI KNN':<20} | {metrics_genai['Accuracy']:.4f} | {metrics_genai['Precision']:.4f} | {metrics_genai['Recall']:.4f} | {metrics_genai['F1']:.4f} | {avg_t_genai:.6f}")
print(f"{'Sklearn KNN':<20} | {metrics_sk['Accuracy']:.4f} | {metrics_sk['Precision']:.4f} | {metrics_sk['Recall']:.4f} | {metrics_sk['F1']:.4f} | {avg_t_sk:.6f}")
print(f"{'Weighted KNN':<20} | {metrics_weighted['Accuracy']:.4f} | {metrics_weighted['Precision']:.4f} | {metrics_weighted['Recall']:.4f} | {metrics_weighted['F1']:.4f} | {avg_t_weighted:.6f}")

# A8 Plot: Accuracy vs K
k_range = range(1, 11)
acc_genai = []
acc_sk = []

for k in k_range:
    # GenAI
    m1 = GenAI_KNN(k=k)
    m1.fit(X_train, y_train)
    acc_genai.append(m1.score(X_test, y_test))
    
    # Sklearn
    m2 = KNeighborsClassifier(n_neighbors=k)
    m2.fit(X_train, y_train)
    acc_sk.append(m2.score(X_test, y_test))

plt.figure(figsize=(10, 5))
plt.plot(k_range, acc_genai, label='GenAI Implementation', marker='o')
plt.plot(k_range, acc_sk, label='Sklearn Package', marker='x')
plt.title('A8: Accuracy vs K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()