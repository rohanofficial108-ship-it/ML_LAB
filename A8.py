import pandas as pd
df = pd.read_csv("project_dataset.csv")
def mean(data):
    return sum(data)/len(data)
def variance(data):
    m = mean(data)
    total = 0
    for i in data:
        total += (i-m)**2
    return total/len(data)
def standard_deviation(data):
    return variance(data)**0.5
numeric_df = df.select_dtypes(include=["int64","float64"])
for column in numeric_df.columns:
    values = numeric_df[column].dropna().values
    print("--------------------------------")
    print(column)
    print("Mean :", mean(values))
    print("Variance :", variance(values))
    print("Std :", standard_deviation(values))