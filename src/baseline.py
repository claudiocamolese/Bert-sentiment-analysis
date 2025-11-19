from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import numpy as np

def get_tfidf_vectors_and_labels(df, split="Train", max_features=1000):
    """Generate TF-IDF vectors and corresponding labels from a dataset split.

    This function filters the dataframe according to the specified split 
    (e.g., "Train" or "Test"), extracts the text sentences and labels, 
    and transforms the sentences into TF-IDF vectors with a maximum 
    number of features.

    Args:
        df (pandas.DataFrame): Dataset containing at least the columns 
            'sentence', 'label', and 'Split'.
        split (str, optional): Subset of the data to use 
            (typically "Train" or "Test"). Defaults to "Train".
        max_features (int, optional): Maximum number of TF-IDF features to keep. 
            Defaults to 1000.

    Returns:
        tuple:
            - numpy.ndarray: TF-IDF matrix of shape (N, max_features).
            - pandas.Series: Corresponding labels.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    texts = df[df.Split==split]['sentence']
    vectors = vectorizer.fit_transform(texts)
    labels = df[df.Split==split]['label']
    
    return vectors.toarray(), labels

def plot_baseline(df_data):
    """Plot baseline F1-scores for increasing TF-IDF vocabulary sizes.

    This function runs the baseline classifier with multiple values of 
    TF-IDF `max_features`, collects both train and test F1-scores, and 
    generates a plot to visualize their trend. The resulting figure is 
    saved locally.

    Args:
        df_data (pandas.DataFrame): The full dataset, already split into 
            training and test subsets (via a 'Split' column).
    """
    f1_scores_train = []
    f1_scores_valid = []
    max_features_array = [5, 10, 50, 100, 200,300,400,500,600,700,800,900,1000]
    for max_features in max_features_array:
        f1_train, f1_valid, acc = baseline(df_data, max_features = max_features)
        f1_scores_train.append(f1_train)
        f1_scores_valid.append(f1_valid)

    plt.plot(np.arange(len(max_features_array)),f1_scores_train,label="F1 Train")
    plt.plot(np.arange(len(max_features_array)),f1_scores_valid,label="F1 Test")
    plt.xticks(np.arange(len(max_features_array)),max_features_array)
    plt.xlabel("Number of Features")
    plt.ylabel("F1 score")
    plt.legend()
    plt.savefig("./plot/F1_scores")

def baseline(df, max_features= 100, model = 'nb'):
    """Train a baseline classifier using TF-IDF features and compute metrics.

    This function builds TF-IDF vectors for both training and test splits, 
    trains either a Naive Bayes or Logistic Regression classifier, and 
    evaluates its performance using macro F1 and accuracy.

    Args:
        df (pandas.DataFrame): Dataset containing 'sentence', 'label', 
            and 'Split' columns.
        max_features (int, optional): Maximum number of TF-IDF features to use. 
            Defaults to 100.
        model (str, optional): Type of classifier to train:
            - 'nb' → Multinomial Naive Bayes
            - otherwise → Logistic Regression
            Defaults to 'nb'.

    Returns:
        tuple:
            - float: Train macro F1-score.
            - float: Test macro F1-score.
            - float: Test accuracy.
    """
    vectors_train, labels_train = get_tfidf_vectors_and_labels(df, split="Trai", max_features = max_features)
    vectors_test, labels_test = get_tfidf_vectors_and_labels(df, split="Test", max_features = max_features)

    if model == 'nb':
      classifier = MultinomialNB().fit(vectors_train, labels_train)
    else:
      classifier = LogisticRegression().fit(vectors_train, labels_train)

    predicted_train = classifier.predict(vectors_train)
    predicted_test = classifier.predict(vectors_test)

    f1_train = f1_score(labels_train, predicted_train, average='macro') 
    f1_test = f1_score(labels_test, predicted_test, average='macro')
    accuracy = (predicted_test == labels_test).sum()/len(predicted_test)

    return f1_train, f1_test, accuracy

