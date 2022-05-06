

import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import nltk
import project3

from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk
from nltk import tree2conlltags
from nltk.corpus import treebank

nltk.download("maxent_ne_chunker")
nltk.download("words")
import spacy

tokenize_words = nltk.word_tokenize

def test_prediction():

    filename = "Unredactor.txt"

    df = project3.create_dataframe(filename)

    df = df.iloc[0:1443]

    df = df.drop(index=[440, 441, 442, 443, 444, 445, 1135, 1136, 1137, 1138, 1139])

    df = project3.swap_columns(df, "Person", "Sentence")

    df = project3.normalize_text(df)

    df = project3.consolidate_names(df)

    final_features = project3.feature_selection(df)

    df.loc[:, "Final_Features"] = final_features

    for i in df.index:
        df["Sentence"][i] = df["Sentence"][i].replace("\u2588", "")

    # print(df.head())

    # print(df.head(100))

    # df.to_csv('datagy.csv')

    # Create a Train DataFrame

    df_train = df.loc[df["Type"] == "training"]

    # df_train.to_csv("datagy1.csv")

    # Create a Validation Data Frame

    df_validation = df.loc[df["Type"] == "validation"]

    # df_validation.to_csv("datagy2.csv")

    # Create a Test Data Frame

    df_test = df.loc[(df["Type"] == "testing") | (df["Type"] == "test")]

    # df_test.to_csv("datagy3.csv")

    # Creation of train, validation, and test datasets

    train_corpus, train_label = (
        np.array(df_train["Final_Features"]),
        np.array(df_train["Person"]),
    )

    # print(train_corpus.shape)

    # print(train_corpus[0])

    # print(len(train_label))

    validation_corpus, validation_label = (
        np.array(df_validation["Final_Features"]),
        np.array(df_validation["Person"]),
    )

    # print(validation_corpus.shape)

    # print(validation_corpus[0])

    # print(len(validation_label))

    test_corpus, test_label = (
        np.array(df_test["Final_Features"]),
        np.array(df_test["Person"]),
    )

    # print(test_corpus.shape)

    # print(test_corpus[0])

    # print(len(test_corpus))

    # Need to do feature engineering first
    # Count Vectorizer

    cv = CountVectorizer(binary=False, min_df=0.0, max_df=1.0)

    cv_train_features = cv.fit_transform(train_corpus)

    cv_validation_features = cv.transform(validation_corpus)

    cv_test_features = cv.transform(test_corpus)

    # TF-IDF Vectorizer

    #tv = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0)

    #tv_train_features = tv.fit_transform(train_corpus)

    #tv_validation_features = tv.transform(validation_corpus)

    #tv_test_features = tv.transform(test_corpus)

    mnb = MultinomialNB(alpha=1)
    mnb.fit(cv_train_features, train_label)

    mnb_bow_validation_score = mnb.score(cv_validation_features, validation_label)

    mnb_bow_test_score = mnb.score(cv_test_features, test_label)

    mnb_test_predictions = mnb.predict(cv_test_features)
    precision = precision_score(test_label, mnb_test_predictions, average="micro")
    recall = recall_score(test_label, mnb_test_predictions, average="micro")
    F1_score = f1_score(test_label, mnb_test_predictions, average="micro")
    accuracy = accuracy_score(test_label, mnb_test_predictions)

    assert F1_score> 0
    assert precision > 0
    assert recall > 0
    
    
