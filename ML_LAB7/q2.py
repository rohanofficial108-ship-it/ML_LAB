import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv('Iris.csv')
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
    return df

df = load_data()

def calculate_gini(target_column):
    counts = target_column.value_counts(normalize=True)
    gini = 1 - np.sum(counts ** 2)
    return gini

gini_val = calculate_gini(df['Species'])
print(f"A2. Gini Index of Species: {gini_val:.4f}")