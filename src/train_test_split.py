import numpy as np

def train_test_split(df, train_size=0.8, random_state=42):
    """
    Crea una colonna 'Split' che assegna 'Train' a train_size della data e 'Test' al resto.
    """
    np.random.seed(random_state)
    # Numero totale di righe
    n = len(df)
    # Indici mischiati
    shuffled_indices = np.random.permutation(n)
    # Numero di righe per il train
    train_count = int(train_size * n)
    
    # Creiamo un array vuoto
    split = np.array(["Test"] * n)
    # Assegniamo 'Train' ai primi train_count indici
    split[shuffled_indices[:train_count]] = "Train"
    
    # Aggiungiamo la colonna al DataFrame
    df['Split'] = split

