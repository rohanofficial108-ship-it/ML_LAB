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

class SimpleDecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def _entropy(self, y):
        return calculate_entropy(y)

    def _information_gain(self, X, y, feature):
        pass

    def fit(self, X, y):
        print("A5. Decision Tree Module initialized and 'fit' called.")
        pass

    def predict(self, X):
        pass

my_tree = SimpleDecisionTree()
my_tree.fit(df.drop('Species', axis=1), df['Species'])