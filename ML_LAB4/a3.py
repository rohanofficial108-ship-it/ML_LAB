# GenAI Tool Used: ChatGPT
# AI assistance was used for code generation.


def minkowski_distance(vector_a, vector_b, p):
    """
    Calculate generalized Minkowski distance.
    """

    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same dimension.")

    if p <= 0:
        raise ValueError("p must be greater than zero.")

    total = 0

    for i in range(len(vector_a)):
        total += abs(vector_a[i] - vector_b[i]) ** p

    return total ** (1 / p)


def main():

    vector_a = [2, 4, 6, 8]
    vector_b = [1, 3, 5, 7]

    for p in [1, 2, 3]:

        distance = minkowski_distance(
            vector_a,
            vector_b,
            p
        )

        print("p =", p)
        print("Minkowski Distance =", distance)

        if p == 1:
            print("Manhattan Distance")

        elif p == 2:
            print("Euclidean Distance")

        print()


if __name__ == "__main__":
    main()