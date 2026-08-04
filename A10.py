import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("project_dataset.csv")
numeric_df = df.select_dtypes(include=["int64","float64"])
feature = numeric_df.columns[0]
values = numeric_df[feature].dropna()
plt.hist(values, bins=10, edgecolor="black")
plt.xlabel(feature)
plt.ylabel("Frequency")
plt.title(feature)
plt.show()
print("Mean :", values.mean())
print("Variance :", values.var())