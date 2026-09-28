from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "target"
ID_COLUMN = "ID_code"
RANDOM_STATE = 888


def load_data(path):
    """
    Load the raw dataset.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the CSV dataset.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            f"Dataset is empty: {path}"
        )

    return df


def create_balanced_sample(
    df,
    target_column=TARGET_COLUMN,
    random_state=RANDOM_STATE,
):
    """
    Create a balanced binary classification sample.

    All positive observations (target == 1) are retained.
    An equal number of negative observations (target == 0)
    are randomly sampled.

    Parameters
    ----------
    df : pandas.DataFrame
        Original dataset.

    target_column : str
        Binary target column.

    random_state : int
        Random seed.

    Returns
    -------
    pandas.DataFrame
        Balanced and shuffled dataset.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            "not found in dataset."
        )

    good = df[
        df[target_column] == 0
    ]

    bad = df[
        df[target_column] == 1
    ]

    if len(bad) == 0:
        raise ValueError(
            "No observations with target == 1."
        )

    if len(good) < len(bad):
        raise ValueError(
            "There are fewer target==0 observations "
            "than target==1 observations."
        )

    n_bads = len(bad)

    good_sample = good.sample(
        n=n_bads,
        random_state=random_state,
    )

    balanced = pd.concat(
        [
            good_sample,
            bad,
        ],
        ignore_index=True,
    )

    balanced = balanced.sample(
        frac=1,
        random_state=random_state,
    ).reset_index(drop=True)

    return balanced


def split_features_target(
    df,
    target_column=TARGET_COLUMN,
    id_column=ID_COLUMN,
):
    """
    Separate features and target.

    ID_code is removed from the feature matrix.

    Parameters
    ----------
    df : pandas.DataFrame
        Balanced dataset.

    target_column : str
        Target column.

    id_column : str
        Identifier column to exclude.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix.

    y : pandas.Series
        Target vector.
    """

    required_columns = {
        target_column,
        id_column,
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{missing_columns}"
        )

    X = df.drop(
        columns=[
            id_column,
            target_column,
        ]
    )

    y = df[target_column]

    return X, y


def create_train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
):
    """
    Create a stratified train/test split.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.

    y : pandas.Series
        Target.

    test_size : float
        Proportion allocated to the test set.

    random_state : int
        Random seed.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """

    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )