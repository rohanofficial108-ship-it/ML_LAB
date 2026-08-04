import pandas as pd
import numpy as np
df = pd.read_csv("project_dataset.csv")
def dot_product(a, b):
    total = 0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total
def euclidean_norm(a):
    total = 0
    for i in a:
        total += i ** 2
    return total ** 0.5
numeric_df = df.select_dtypes(include=["int64","float64"])
A = numeric_df.iloc[0].values
B = numeric_df.iloc[1].values
print("Custom Dot Product :", dot_product(A,B))
print("Numpy Dot Product :", np.dot(A,B))
print()
print("Custom Norm :", euclidean_norm(A))
print("Numpy Norm :", np.linalg.norm(A))