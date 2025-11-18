from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import numpy as np

def get_tfidf_vectors_and_labels(df, split="Train", max_features=1000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    
    # Fit e transform sulle frasi
    texts = df[df.Split==split]['sentence']
    vectors = vectorizer.fit_transform(texts)
    
    # Label corrispondenti
    labels = df[df.Split==split]['label']
    
    return vectors.toarray(), labels

def plot_baseline(df_data):
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
    vectors_train, labels_train = get_tfidf_vectors_and_labels(df, split="Trai", max_features = max_features)
    vectors_test, labels_test = get_tfidf_vectors_and_labels(df, split="Test", max_features = max_features)

    if model == 'nb':
      # train a multinomial Naive-Bayes classifier on the tfidf vectors
      classifier = MultinomialNB().fit(vectors_train, labels_train)
    else:
      ## Here is another classifier:
      classifier = LogisticRegression().fit(vectors_train, labels_train)

    # use classifier to predict the labels of the test set
    predicted_train = classifier.predict(vectors_train)
    predicted_test = classifier.predict(vectors_test)

    f1_train = f1_score(labels_train, predicted_train, average='macro') #or weighted if you'd like!
    f1_test = f1_score(labels_test, predicted_test, average='macro') #or weighted if you'd like!
    accuracy = (predicted_test == labels_test).sum()/len(predicted_test)

    return f1_train, f1_test, accuracy

