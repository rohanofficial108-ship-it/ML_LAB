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

def calculate_information_gain(df, feature_name, target_name, bins=4):
    total_entropy = calculate_entropy(df[target_name])
    if pd.api.types.is_numeric_dtype(df[feature_name]):
        feature_data = pd.cut(df[feature_name], bins=bins, labels=False)
    else:
        feature_data = df[feature_name]
    unique_values = feature_data.unique()
    weighted_entropy = 0
    for val in unique_values:
        subset = df[feature_data == val]
        prob = len(subset) / len(df)
        subset_entropy = calculate_entropy(subset[target_name])
        weighted_entropy += prob * subset_entropy
    information_gain = total_entropy - weighted_entropy
    return information_gain

def find_root_node(df, target_name):
    features = df.columns.drop(target_name)
    ig_scores = {}
    for feature in features:
        ig = calculate_information_gain(df, feature, target_name)
        ig_scores[feature] = ig
        print(f"Feature: {feature}, IG: {ig:.4f}")
    best_feature = max(ig_scores, key=ig_scores.get)
    print(f"Best Feature for Root Node: {best_feature}")
    return best_feature

best_root = find_root_node(df, 'Species')