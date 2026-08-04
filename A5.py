import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("project_dataset.csv")
def minkowski_distance(vector1, vector2, p):
    distance = 0
    for i in range(len(vector1)):
        distance += abs(vector1[i] - vector2[i]) ** p
    return distance ** (1 / p)
numeric_df = df.select_dtypes(include=["int64","float64"])
vector1 = numeric_df.iloc[0].values
vector2 = numeric_df.iloc[1].values
distance = []
for p in range(1,11):
    distance.append(minkowski_distance(vector1, vector2, p))
plt.plot(range(1,11), distance, marker="o")
plt.xlabel("p")
plt.ylabel("Distance")
plt.title("Minkowski Distance")
plt.grid(True)
plt.show()