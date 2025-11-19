import pandas as pd

def load_data(filename, classes):
    """Build the dataset with sentences and their ground truth

    Args:
        filename (str): path of the file
        classes (list): list of the possible classification

    Returns:
        pd.DataFrame: pandas datafram with sentences and ground truth
    """
    
    data = []
    with open(filename, encoding='iso-8859-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if '@' in line:
                sentence, label = line.split('@')
                sentence = sentence.strip()
                label = label.strip()
                if label in classes:
                    data.append([sentence, label])
    
    df = pd.DataFrame(data, columns=['sentence', 'label'])
    return df

if __name__ == "__main__":

    filename = "./data/Sentences_75Agree.txt"
    classes = ['negative', 'neutral', 'positive']

    df_data = load_data(filename, classes)
    print(df_data.head())
    
    
