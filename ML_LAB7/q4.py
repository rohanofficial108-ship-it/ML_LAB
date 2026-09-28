import pandas as pd

def load_data():
    df = pd.read_csv('Iris.csv')
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
    return df

df = load_data()

def bin_feature(series, num_bins=4, strategy='equal_width'):
    if strategy == 'equal_width':
        return pd.cut(series, bins=num_bins, labels=False)
    elif strategy == 'frequency':
        return pd.qcut(series, q=num_bins, labels=False, duplicates='drop')
    else:
        raise ValueError("Strategy must be 'equal_width' or 'frequency'")

binned_default = bin_feature(df['SepalWidthCm'])
print(f"A4. Default Binning (First 5): \n{binned_default.head()}")

binned_freq = bin_feature(df['SepalWidthCm'], num_bins=3, strategy='frequency')
print(f"A4. Frequency Binning (First 5): \n{binned_freq.head()}")