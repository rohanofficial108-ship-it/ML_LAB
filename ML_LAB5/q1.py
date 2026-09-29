import csv
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

# --- A1:

class MyKNN_Manual:
    def __init__(self, k=3, distance_metric='euclidean', sort_algo='bubble', weighted=False):
        self.k = k
        self.distance_metric = distance_metric
        self.sort_algo = sort_algo
        self.weighted = weighted
        
    # A1a. Encoding
    def _encode(self, label):
        mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
        return mapping.get(label, -1)

    # A1b. Data Imputation 
    def _impute(self, data):
        pass 

    # A1c. Distance Calculation
    def _calculate_distance(self, row1, row2):
        if self.distance_metric == 'euclidean':
            dist = 0
            for i in range(len(row1)):
                dist += (row1[i] - row2[i]) ** 2
            return math.sqrt(dist)
        elif self.distance_metric == 'manhattan':
            return sum(abs(a - b) for a, b in zip(row1, row2))
        return 0

    # A1d. Sorting Algorithms
    def _sort_neighbors(self, neighbors):
        # neighbors is list of tuples (distance, label)
        if self.sort_algo == 'bubble':
            n = len(neighbors)
            for i in range(n):
                for j in range(0, n-i-1):
                    if neighbors[j][0] > neighbors[j+1][0]:
                        neighbors[j], neighbors[j+1] = neighbors[j+1], neighbors[j]
            return neighbors
        elif self.sort_algo == 'insertion':
            for i in range(1, len(neighbors)):
                key = neighbors[i]
                j = i-1
                while j >= 0 and key[0] < neighbors[j][0]:
                    neighbors[j+1] = neighbors[j]
                    j -= 1
                neighbors[j+1] = key
            return neighbors
        else: # Default Python sort (Timsort)
            return sorted(neighbors, key=lambda x: x[0])

    # A1e & A1f. Identify Neighbors & Class Evaluation
    def _predict_single(self, train_data, train_labels, test_row):
        distances = []
        for i in range(len(train_data)):
            d = self._calculate_distance(test_row, train_data[i])
            distances.append((d, train_labels[i]))
        
        # Sort
        sorted_neighbors = self._sort_neighbors(distances)
        
        # Identify k-nearest
        k_nearest = sorted_neighbors[:self.k]
        
        # Voting
        if self.weighted:
            # A2: Weighted kNN (Weight = 1/d)
            votes = {}
            for d, label in k_nearest:
                weight = 1 / (d + 1e-5) # avoid div by zero
                votes[label] = votes.get(label, 0) + weight
            # Tie breaking: highest weight wins
            return max(votes, key=votes.get)
        else:
            # Majority Voting
            labels = [label for _, label in k_nearest]
            count = Counter(labels)
            # Tie breaking: smallest class label wins (arbitrary, but consistent)
            return min(count, key=lambda k: (-count[k], k))

    # A7. Fit, Predict, Score
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            predictions.append(self._predict_single(self.X_train, self.y_train, row))
        return predictions

    def score(self, X_test, y_test):
        preds = self.predict(X_test)
        correct = sum(1 for p, t in zip(preds, y_test) if p == t)
        return correct / len(y_test)

#Data Loading & Pre
def load_data(filename):
    X, y = [], []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            # Using SepalLength, SepalWidth, PetalLength, PetalWidth
            features = [float(x) for x in row[1:5]]
            label = row[5]
            X.append(features)
            y.append(label)
    return X, y



# Load Data
X_raw, y_raw = load_data('Iris.csv')

# A3. Split Data 

X_bin, y_bin = [], []
for x, y in zip(X_raw, y_raw):
    if y in ['Iris-setosa', 'Iris-versicolor']:
        X_bin.append(x)
        y_bin.append(y)

X_train, X_test, y_train, y_test = train_test_split(X_bin, y_bin, test_size=0.3, random_state=42)

# A4/A5/A6: Sklearn Implementation for Comparison
sk_neigh = KNeighborsClassifier(n_neighbors=3)
sk_neigh.fit(X_train, y_train)
sk_preds = sk_neigh.predict(X_test)
sk_acc = sk_neigh.score(X_test, y_test)

# Custom Implementation
my_knn = MyKNN_Manual(k=3, weighted=False)
my_knn.fit(X_train, y_train)
my_preds = my_knn.predict(X_test)
my_acc = my_knn.score(X_test, y_test)

# A8: Comparative Analysis (Range of K)
k_values = [1, 3, 5, 7, 9]
my_accuracies = []
sk_accuracies = []

for k in k_values:
    # Custom
    mk = MyKNN_Manual(k=k)
    mk.fit(X_train, y_train)
    my_accuracies.append(mk.score(X_test, y_test))
    
    # Sklearn
    sk = KNeighborsClassifier(n_neighbors=k)
    sk.fit(X_train, y_train)
    sk_accuracies.append(sk.score(X_test, y_test))

# Plotting A8
plt.figure(figsize=(10, 6))
plt.plot(k_values, my_accuracies, marker='o', label='Custom KNN')
plt.plot(k_values, sk_accuracies, marker='s', label='Sklearn KNN')
plt.title('Accuracy vs K Value (Custom vs Sklearn)')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# A9: Weighted kNN Experiment
weighted_accuracies = []
for k in k_values:
    wk = MyKNN_Manual(k=k, weighted=True)
    wk.fit(X_train, y_train)
    weighted_accuracies.append(wk.score(X_test, y_test))





def measure_time(model, X_train, y_train, X_test, is_sklearn=False):
    start = time.time()
    if is_sklearn:
        model.fit(X_train, y_train)
        model.predict(X_test)
    else:
        model.fit(X_train, y_train)
        model.predict(X_test)
    return time.time() - start

times_my = []
times_sk = []
times_weighted = []

for _ in range(10):
    times_my.append(measure_time(MyKNN_Manual(k=3), X_train, y_train, X_test))
    times_sk.append(measure_time(KNeighborsClassifier(n_neighbors=3), X_train, y_train, X_test, True))
    times_weighted.append(measure_time(MyKNN_Manual(k=3, weighted=True), X_train, y_train, X_test))

avg_time_my = sum(times_my) / 10
avg_time_sk = sum(times_sk) / 10
avg_time_weighted = sum(times_weighted) / 10

# Metrics Calculation 

print(f"{'Method':<20} | {'Accuracy':<10} | {'Avg Time (s)':<15}")
print("-" * 50)
print(f"{'Custom KNN':<20} | {my_accuracies[1]:<10.4f} | {avg_time_my:<15.6f}")
print(f"{'Sklearn KNN':<20} | {sk_accuracies[1]:<10.4f} | {avg_time_sk:<15.6f}")
print(f"{'Weighted KNN':<20} | {weighted_accuracies[1]:<10.4f} | {avg_time_weighted:<15.6f}")