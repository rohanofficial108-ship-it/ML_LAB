import pandas as pd
import matplotlib.pyplot as plt

# GenAI Tool Used: ChatGPT
# AI assistance was used for code generation.


def minkowski_distance(vector_a, vector_b, p):

    total = 0

    for i in range(len(vector_a)):
        total += abs(vector_a[i] - vector_b[i]) ** p

    return total ** (1 / p)


def load_numeric_data():

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

    return numeric_dataframe


def calculate_distances(vector_a, vector_b):

    p_values = []
    distances = []

    for p in range(1, 11):

        distance = minkowski_distance(
            vector_a,
            vector_b,
            p
        )

        p_values.append(p)
        distances.append(distance)

    return p_values, distances


def plot_distances(p_values, distances):

    plt.plot(
        p_values,
        distances,
        marker="o"
    )

    plt.xlabel("p")
    plt.ylabel("Minkowski Distance")

    plt.title(
        "Minkowski Distance for p = 1 to 10"
    )

    plt.grid(True)
    plt.show()


def main():

    dataframe = load_numeric_data()

    vector_a = dataframe.iloc[0].tolist()
    vector_b = dataframe.iloc[1].tolist()

    p_values, distances = calculate_distances(
        vector_a,
        vector_b
    )

    for p, distance in zip(
        p_values,
        distances
    ):

        print(
            "p =",
            p,
            "Distance =",
            distance
        )

    plot_distances(
        p_values,
        distances
    )


if __name__ == "__main__":
    main()