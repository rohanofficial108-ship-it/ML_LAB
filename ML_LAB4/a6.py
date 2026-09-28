import pandas as pd
import numpy as np
import math

# GenAI Tool Used: ChatGPT
# AI assistance was used for code generation.


def dot_product(vector_a, vector_b):

    if len(vector_a) != len(vector_b):
        raise ValueError(
            "Vectors must have the same dimension."
        )

    result = 0

    for i in range(len(vector_a)):
        result += vector_a[i] * vector_b[i]

    return result


def euclidean_norm(vector):

    total = 0

    for value in vector:
        total += value ** 2

    return math.sqrt(total)


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

    vector_a = numeric_dataframe.iloc[0].tolist()
    vector_b = numeric_dataframe.iloc[1].tolist()

    own_dot = dot_product(
        vector_a,
        vector_b
    )

    numpy_dot = np.dot(
        vector_a,
        vector_b
    )

    own_norm_a = euclidean_norm(vector_a)
    own_norm_b = euclidean_norm(vector_b)

    numpy_norm_a = np.linalg.norm(vector_a)
    numpy_norm_b = np.linalg.norm(vector_b)

    print("Dot Product")
    print("Own Function:", own_dot)
    print("NumPy:", numpy_dot)

    print("\nVector A Norm")
    print("Own Function:", own_norm_a)
    print("NumPy:", numpy_norm_a)

    print("\nVector B Norm")
    print("Own Function:", own_norm_b)
    print("NumPy:", numpy_norm_b)


if __name__ == "__main__":
    main()