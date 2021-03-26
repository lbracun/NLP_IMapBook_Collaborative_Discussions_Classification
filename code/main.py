import pandas as pd

DISCUSSIONS_DATASET_PATH = "../data/IMapBook_discussions_dataset.xlsx"


def main():
    df: pd.DataFrame = pd.read_excel(DISCUSSIONS_DATASET_PATH)
    print(df.head())


if __name__ == "__main__":
    main()
