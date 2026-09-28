import unittest
import numpy as np

# GenAI Tool Used: ChatGPT
# AI assistance was used for generating unit-test cases.
# Tests cover functions from the Lab 03 / Lab 04 exercises.


def minkowski_distance(a, b, p):

    total = 0

    for i in range(len(a)):
        total += abs(a[i] - b[i]) ** p

    return total ** (1 / p)


def dot_product(a, b):

    result = 0

    for i in range(len(a)):
        result += a[i] * b[i]

    return result


def euclidean_norm(vector):

    total = 0

    for value in vector:
        total += value ** 2

    return total ** 0.5


def calculate_mean(data):

    return sum(data) / len(data)


def calculate_variance(data):

    mean = calculate_mean(data)

    return sum(
        (x - mean) ** 2
        for x in data
    ) / len(data)


def calculate_std(data):

    return calculate_variance(data) ** 0.5


class TestLabFunctions(unittest.TestCase):

    def test_minkowski_manhattan(self):

        result = minkowski_distance(
            [1, 2, 3],
            [2, 4, 6],
            1
        )

        self.assertEqual(
            result,
            6
        )

    def test_minkowski_euclidean(self):

        result = minkowski_distance(
            [0, 0],
            [3, 4],
            2
        )

        self.assertEqual(
            result,
            5
        )

    def test_dot_product(self):

        result = dot_product(
            [1, 2, 3],
            [4, 5, 6]
        )

        self.assertEqual(
            result,
            32
        )

    def test_euclidean_norm(self):

        result = euclidean_norm(
            [3, 4]
        )

        self.assertEqual(
            result,
            5
        )

    def test_mean(self):

        result = calculate_mean(
            [10, 20, 30]
        )

        self.assertEqual(
            result,
            20
        )

    def test_variance(self):

        result = calculate_variance(
            [10, 20, 30]
        )

        self.assertAlmostEqual(
            result,
            66.6666666667
        )

    def test_standard_deviation(self):

        result = calculate_std(
            [3, 3, 3]
        )

        self.assertEqual(
            result,
            0
        )


if __name__ == "__main__":

    unittest.main()