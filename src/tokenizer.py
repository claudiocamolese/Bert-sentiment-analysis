from transformers import DistilBertTokenizer
from matplotlib import pyplot as plt
import numpy as np

def tokenizer_processing(model, df_data):
    tokenizer = DistilBertTokenizer.from_pretrained(model, padding=True, truncation=True)
    # let's check out how the tokenizer works
    all_texts = df_data['sentence'].tolist()

    # Calcola la lunghezza in token di ciascuna frase
    lengths = [len(tokenizer.encode(text, add_special_tokens=True)) for text in all_texts]

    # Statistiche
    print(f"Lunghezza media: {np.mean(lengths):.1f}")
    print(f"Lunghezza mediana: {np.median(lengths):.1f}")
    print(f"Lunghezza massima: {np.max(lengths)}")

    # Istogramma delle lunghezze
    plt.figure(figsize=(10,5))
    plt.hist(lengths, bins=50, color='skyblue', edgecolor='black')
    plt.title("Distribuzione lunghezze delle frasi in token")
    plt.xlabel("Numero di token")
    plt.ylabel("Frequenza")
    plt.savefig("./plot/freq_tok")
    
    return tokenizer
