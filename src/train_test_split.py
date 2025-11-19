import numpy as np

def train_test_split(df, train_size=0.8, random_state=42):
    """
    Randomly splits a DataFrame into train and test subsets by adding
    a new column named ``'Split'`` with values ``'Train'`` or ``'Test'``.

    The function performs a shuffled split using NumPy permutation and does
    not return a new DataFrame: it modifies the input DataFrame in place.

    Args:
        df (pandas.DataFrame): 
            The dataset to be split. The function will add (or overwrite)
            the column ``df['Split']``.
        train_size (float, optional): 
            Proportion of samples to assign to the training subset.
            Must be between 0 and 1. Defaults to 0.8.
        random_state (int, optional): 
            Seed for the NumPy random number generator to ensure
            reproducibility. Defaults to 42.

    Returns:
        None: 
            The function operates in place. The input DataFrame will contain
            a new column ``'Split'`` with values ``'Train'`` or ``'Test'``.
    """
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(df))
    train_count = int(train_size * len(df))
    split = np.array(["Test"] * len(df))
    split[shuffled_indices[:train_count]] = "Train"
    df['Split'] = split
