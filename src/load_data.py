import pandas as pd

def load_data(filename, classes):
    data = []
    # Apriamo il file con la codifica iso-8859-1
    with open(filename, encoding='iso-8859-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Separiamo la frase dalla label usando '@' come separatore
            if '@' in line:
                sentence, label = line.split('@')
                sentence = sentence.strip()
                label = label.strip()
                if label in classes:
                    data.append([sentence, label])
    
    # Creiamo il DataFrame
    df = pd.DataFrame(data, columns=['sentence', 'label'])
    return df

if __name__ == "__main__":

    filename = "./data/Sentences_75Agree.txt"
    classes = ['negative', 'neutral', 'positive']

    df_data = load_data(filename, classes)
    print(df_data.head())
    
    
