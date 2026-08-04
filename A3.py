import pandas as pd
df = pd.read_csv("project_dataset.csv")
def one_hot_encoding(dataframe):

    return pd.get_dummies(dataframe)
encoded_df = one_hot_encoding(df)
print(encoded_df.head())
print("Original Shape :", df.shape)
print("Encoded Shape :", encoded_df.shape)