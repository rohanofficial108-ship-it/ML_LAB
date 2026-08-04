import pandas as pd
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("project_dataset.csv")
def label_encoding(dataframe):
    data = dataframe.copy()
    encoder = LabelEncoder()
    for column in data.columns:

        if data[column].dtype == "object":

            data[column] = encoder.fit_transform(data[column].astype(str))
    return data
encoded_df = label_encoding(df)
print(encoded_df.head())