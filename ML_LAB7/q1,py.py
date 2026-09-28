import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv('Iris.csv')
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
    return df

df = load_data()

def calculate_entropy(target_column, bins=4):
    if pd.api.types.is_numeric_dtype(target_column):
        binned_data = pd.cut(target_column, bins=bins, labels=False)
        counts = binned_data.value_counts(normalize=True)
    else:
        counts = target_column.value_counts(normalize=True)
    entropy = -np.sum(counts * np.log2(counts))
    return entropy

entropy_continuous = calculate_entropy(df['PetalLengthCm'])
print(f"A1. Entropy of continuous PetalLengthCm (binned): {entropy_continuous:.4f}")

entropy_categorical = calculate_entropy(df['Species'])
print(f"A1. Entropy of Species: {entropy_categorical:.4f}")