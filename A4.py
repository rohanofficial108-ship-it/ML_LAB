import pandas as pd
df = pd.read_csv("project_dataset.csv")
def minkowski_distance(vector1, vector2, p):
    distance = 0
    for i in range(len(vector1)):
        distance += abs(vector1[i] - vector2[i]) ** p
    return distance ** (1 / p)
numeric_df = df.select_dtypes(include=["int64","float64"])
vector1 = numeric_df.iloc[0].values
vector2 = numeric_df.iloc[1].values
print("Manhattan Distance :", minkowski_distance(vector1, vector2, 1))
print("Euclidean Distance :", minkowski_distance(vector1, vector2, 2))