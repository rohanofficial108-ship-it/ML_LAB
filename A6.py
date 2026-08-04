import pandas as pd
from scipy.spatial.distance import minkowski
df = pd.read_csv("project_dataset.csv")
def my_distance(vector1, vector2, p):
    distance = 0
    for i in range(len(vector1)):
        distance += abs(vector1[i] - vector2[i]) ** p
    return distance ** (1 / p)
numeric_df = df.select_dtypes(include=["int64","float64"])
vector1 = numeric_df.iloc[0].values
vector2 = numeric_df.iloc[1].values
print("Custom :", my_distance(vector1, vector2, 2))
print("SciPy :", minkowski(vector1, vector2, 2))