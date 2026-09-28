import pandas as pd

# GenAI Tool Used: ChatGPT
# AI assistance was used for code generation.
# The modular structure follows the student's Lab 03 implementation.
#A3


def label_encode(dataframe, column):

    unique_values = dataframe[column].dropna().unique()

    mapping = {}

    for index, value in enumerate(unique_values):
        mapping[value] = index

    encoded = []

    for value in dataframe[column]:
        encoded.append(mapping.get(value, -1))

    return encoded


def one_hot_encode(dataframe, column):

    unique_values = dataframe[column].dropna().unique()

    result = pd.DataFrame(index=dataframe.index)

    for value in unique_values:

        encoded_column = []

        for item in dataframe[column]:

            if item == value:
                encoded_column.append(1)
            else:
                encoded_column.append(0)

        result[column + "_" + str(value)] = encoded_column

    return result


def create_encoded_datasets(dataframe):

    # Label encoded dataset
    label_dataframe = dataframe.copy()

    label_dataframe["Education"] = label_encode(
        label_dataframe,
        "Education"
    )

    label_dataframe["Marital_Status"] = label_encode(
        label_dataframe,
        "Marital_Status"
    )

    # One-hot encoded dataset
    one_hot_dataframe = dataframe.copy()

    education = one_hot_encode(
        dataframe,
        "Education"
    )

    marital = one_hot_encode(
        dataframe,
        "Marital_Status"
    )

    one_hot_dataframe = one_hot_dataframe.drop(
        ["Education", "Marital_Status"],
        axis=1
    )

    one_hot_dataframe = pd.concat(
        [one_hot_dataframe, education, marital],
        axis=1
    )

    return label_dataframe, one_hot_dataframe


def main():

    dataframe = pd.read_excel(
        "Lab Session Data (1).xlsx",
        sheet_name="marketing_campaign"
    )

    label_dataframe, one_hot_dataframe = create_encoded_datasets(
        dataframe
    )

    print("Original Dataset:")
    print(dataframe.shape)

    print("\nAfter Label Encoding:")
    print(label_dataframe.shape)

    print("\nAfter One-Hot Encoding:")
    print(one_hot_dataframe.shape)

    print("\nDimensionality Change:")
    print(
        "Original features:",
        dataframe.shape[1]
    )

    print(
        "Label encoded features:",
        label_dataframe.shape[1]
    )

    print(
        "One-hot encoded features:",
        one_hot_dataframe.shape[1]
    )


if __name__ == "__main__":
    main()