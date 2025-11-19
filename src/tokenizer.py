from transformers import DistilBertTokenizer
from matplotlib import pyplot as plt
import numpy as np

def tokenizer_processing(model, df_data):
    """Computes the tokenizer using the specified model 

    Args:
        model (str): model of the hugging face library to upload
        df_data (pandas dataframe): dataset

    Returns:
        tokenizer: tokenizer model
    """
    tokenizer = DistilBertTokenizer.from_pretrained(model, padding=True, truncation=True)
    all_texts = df_data['sentence'].tolist()
    lengths = [len(tokenizer.encode(text, add_special_tokens=True)) for text in all_texts]
    
    print(f"average lenght: {np.mean(lengths):.1f}")
    print(f"median lenght: {np.median(lengths):.1f}")
    print(f"max lenght: {np.max(lengths)}")
    
    plt.figure(figsize=(10,5))
    plt.hist(lengths, bins=50, color='skyblue', edgecolor='black')
    plt.title("Lenght Distribution of sentences in tokens")
    plt.xlabel("Number of tokens")
    plt.ylabel("Frequency")
    plt.savefig("./plot/freq_tok")
    
    return tokenizer
