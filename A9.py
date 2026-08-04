import pandas as pd
import numpy as np
df = pd.read_csv("project_dataset.csv")
def mean(data):
    return sum(data)/len(data)
def standard_deviation(data):
    m = mean(data)
    total = 0
    for i in data:
        total += (i-m)**2
    return (total/len(data))**0.5
numeric_df = df.select_dtypes(include=["int64","float64"])
for column in numeric_df.columns:
    values = numeric_df[column].dropna().values
    print("--------------------------------")
    print(column)
    print("Custom Mean :", mean(values))
    print("NumPy Mean :", np.mean(values))
    print("Custom Std :", standard_deviation(values))
    print("NumPy Std :", np.std(values))