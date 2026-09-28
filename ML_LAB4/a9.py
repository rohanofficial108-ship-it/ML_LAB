import pandas as pd
import numpy as np

# GenAI Tool Used: ChatGPT
# AI assistance was used for code generation.


def calculate_mean(data):

    total = 0

    for value in data:
        total += value

    return total / len(data)


def calculate_variance(data):

    mean = calculate_mean(data)

    total = 0

    for value in data:
        total += (value - mean) ** 2

    return total / len(data)


def calculate_std(data):

    return calculate_variance(data) ** 0.5


def main():

    dataframe = pd.read_excel(
        "Lab Session Data (1).xlsx",
        sheet_name="marketing_campaign"
    )

    numeric_dataframe = dataframe.select_dtypes(
        include=["int64", "float64"]
    )

    numeric_dataframe = numeric_dataframe.fillna(
        numeric_dataframe.mean()
    )

    print(
        f"{'Feature':<25}"
        f"{'Own Mean':<15}"
        f"{'NumPy Mean':<15}"
        f"{'Own Std':<15}"
        f"{'NumPy Std'}"
    )

    print("-" * 85)

    for column in numeric_dataframe.columns:

        values = numeric_dataframe[column].tolist()

        own_mean = calculate_mean(values)
        own_std = calculate_std(values)

        numpy_mean = np.mean(values)
        numpy_std = np.std(values)

        print(
            f"{column:<25}"
            f"{own_mean:<15.4f}"
            f"{numpy_mean:<15.4f}"
            f"{own_std:<15.4f}"
            f"{numpy_std:.4f}"
        )


if __name__ == "__main__":
    main()