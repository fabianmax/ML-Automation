import pandas as pd


def load_data(path_train, path_test):

    # Read data
    df_train = pd.read_csv(path_train)
    df_test = pd.read_csv(path_test)

    # Columns
    cols_train = df_train.columns.tolist()
    cols_test = df_test.columns.tolist()

    # Subset relevant columns in data
    use_this_cols = set(cols_train).intersection(cols_test)
    df_train = df_train.loc[:, use_this_cols]
    df_test = df_test.loc[:, use_this_cols]

    # Target and features
    y_train = df_train.loc[:, "label"]
    X_train = df_train.drop("label", axis=1)

    y_test = df_test.loc[:, "label"]
    X_test = df_test.drop("label", axis=1)

    return X_train, y_train, X_test, y_test

