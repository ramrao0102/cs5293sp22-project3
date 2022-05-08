# This is Ramkishore Rao's Project3 for CS5293

import urllib.request
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


def download_file(download_url, filename):
    response = urllib.request.urlopen(download_url)
    file = open(filename, "wb")
    file.write(response.read())
    file.close()


def create_dataframe(filename):
    oolumn_names = []

    oolumn_names.append("Name")
    oolumn_names.append("Type")
    oolumn_names.append("Person")
    oolumn_names.append("Sentence")

    df = pd.read_csv(
        filename, sep="\t", names=oolumn_names, on_bad_lines="skip", engine="python"
    )

    return df


def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df


def normalize_text(df):

    for i in df.index:

        # df["Sentence"][i] = df["Sentence"][i].replace("\u2588", "")
        df["Sentence"][i] = df["Sentence"][i].replace(",", "")
        df["Sentence"][i] = df["Sentence"][i].replace("<br /><br", "")
        df["Sentence"][i] = df["Sentence"][i].replace("/>", "")
        df["Sentence"][i] = df["Sentence"][i].replace("\n", "")
        df["Sentence"][i] = df["Sentence"][i].lower()
        df["Sentence"][i] = df["Sentence"][i].replace("'s", "")

        df["Person"][i] = df["Person"][i].replace(r"\s{2,}", " ")
        df["Person"][i] = df["Person"][i].lower()
        df["Person"][i] = df["Person"][i].replace(r"\s$", "")
        df["Person"][i] = df["Person"][i].replace("'s", "")
        df["Person"][i] = df["Person"][i].replace("'$", "")
        df["Person"][i] = df["Person"][i].replace("ms ", "")
        df["Person"][i] = df["Person"][i].replace("young sadako", "sadako")
        df["Person"][i] = df["Person"][i].replace("peter parker", "tobey macguire")
        df["Person"][i] = df["Person"][i].replace("simon cowell", "cowell")
        df["Person"][i] = df["Person"][i].replace("Simon", "cowell")
        df["Person"][i] = df["Person"][i].replace("Jenny", "Jenny Latour")
        df["Person"][i] = df["Person"][i].replace("dolittle", "eddie murphy")

        # print(df["Sentence"][i], df["Person"][i])

    return df


# this below function consolidates context_strings


def consolidate_names(df):

    for i in df.index:
        for j in df.index:

            if "peter" not in df["Person"][i]:
                if "simon" not in df["Person"][i]:
                    if "Christopher" not in df["Person"][i]:
                        if df["Person"][i] in df["Person"][j]:

                            df["Person"][i] = df["Person"][j]

    return df


def feature_selection(df):

    final_features = []

    for j in df.index:
        words = tokenize_words(df["Sentence"][j])

        words_to_left = []
        words_to_right = []
        sent_feature = []
        for i in range(len(words)):
            if "\u2588" in words[i]:
                # print(i, ":", words[i])
                if i >= 8:
                    if i < (len(words) - 10):
                        words_to_left.append(words[i - 8 : i])
                        words_to_right.append(words[i + 1 : i + 9])
                    elif i >= (len(words) - 10) and i < (len(words) - 1):
                        words_to_left.append(words[i - 8 : i])
                        words_to_right.append(words[i + 1 : len(words) - 1])
                    elif i == (len(words) - 1):
                        words_to_left.append(words[i - 8 : i])
                elif i < 8:
                    if i < (len(words) - 10):
                        words_to_left.append(words[0:i])
                        words_to_right.append(words[i + 1 : i + 9])
                    elif i >= (len(words) - 10) and i < (len(words) - 1):
                        words_to_left.append(words[0:i])
                        words_to_right.append(words[i + 1 : len(words) - 1])
                    elif i == (len(words) - 1):
                        words_to_left.append(words[0:i])

        sent_feature = words_to_left + words_to_right

        flat_sent = []
        for i in sent_feature:
            for j in i:
                flat_sent.append(j)

        final_sent_features = " ".join(flat_sent)
        # print(final_sent_features)
        final_features.append(final_sent_features)

    # print(len(final_features))
    return final_features


if __name__ == "__main__":

    path = "https://raw.githubusercontent.com/cegme/cs5293sp22/main/unredactor.tsv"

    filename = "Unredactor.txt"

    download_file(path, filename)

    df = create_dataframe(filename)

    # adjust line below as needed
    df = df.iloc[0:2054]

    # there seem to be some bad lines in her

    df = df.dropna()

    # there seem to be some bad lines in her

    # print(df.head())

    # print(len(df))

    df = swap_columns(df, "Person", "Sentence")

    df = normalize_text(df)

    df = consolidate_names(df)

    final_features = feature_selection(df)

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
    # print(mnb_bow_validation_score)
    # print(mnb_bow_test_score)

    mnb_validation_predictions = mnb.predict(cv_validation_features)
    precision = precision_score(
        validation_label, mnb_validation_predictions, average="micro"
    )
    recall = recall_score(validation_label, mnb_validation_predictions, average="micro")
    F1_score = f1_score(validation_label, mnb_validation_predictions, average="micro")
    accuracy = accuracy_score(validation_label, mnb_validation_predictions)
    print("Precision_Val_Bayes",",",  "Recall_Val_Bayes", ",", "F1_score_Val_Bayes", ",", "Accuracy_Val_Bayes") 

    print(precision, recall, F1_score, accuracy)

    mnb_test_predictions = mnb.predict(cv_test_features)
    precision = precision_score(test_label, mnb_test_predictions, average="micro")
    recall = recall_score(test_label, mnb_test_predictions, average="micro")
    F1_score = f1_score(test_label, mnb_test_predictions, average="micro")
    accuracy = accuracy_score(test_label, mnb_test_predictions)
    print("Precision_Test_Bayes", ",", "Recall_Test_Bayes", ",", "F1_score_Test_Bayes", ",", "Accuracy_Test_Bayes")

    print(precision, recall, F1_score, accuracy)

    # Logistic Regression

    lr = LogisticRegression(penalty="l2", max_iter=100, C=1, random_state=42)
    lr.fit(cv_train_features, train_label)

    lr_bow_validation_score = lr.score(cv_validation_features, validation_label)
    # print(lr_bow_validation_score)
    lr_bow_test_score = lr.score(cv_test_features, test_label)
    # print(lr_bow_test_score)

    lr_validation_predictions = lr.predict(cv_validation_features)
    precision = precision_score(
        validation_label, lr_validation_predictions, average="micro"
    )
    recall = recall_score(validation_label, lr_validation_predictions, average="micro")
    F1_score = f1_score(validation_label, lr_validation_predictions, average="micro")
    accuracy = accuracy_score(validation_label, lr_validation_predictions)

    print("Precision_Val_LR",",",  "Recall_Val_LR", ",", "F1_score_Val_LR", ",", "Accuracy_Val_LR") 
    print(precision, recall, F1_score, accuracy)

    lr_test_predictions = lr.predict(cv_test_features)
    precision = precision_score(test_label, lr_test_predictions, average="micro")
    recall = recall_score(test_label, lr_test_predictions, average="micro")
    F1_score = f1_score(test_label, lr_test_predictions, average="micro")
    accuracy = accuracy_score(test_label, lr_test_predictions)

    print("Precision_Test_LR",",",  "Recall_Test_LR", ",", "F1_score_Test_LR", ",", "Accuracy_Test_LR")
    print(precision, recall, F1_score, accuracy)

    # Support Vector Machines

    svm = LinearSVC(penalty="l2", C=1, random_state=42)
    svm.fit(cv_train_features, train_label)
    svm_bow_validation_score = svm.score(cv_validation_features, validation_label)
    svm_bow_test_score = svm.score(cv_test_features, test_label)
    # print(svm_bow_validation_score)
    # print(svm_bow_test_score)

    svm_validation_predictions = svm.predict(cv_validation_features)
    precision = precision_score(
        validation_label, svm_validation_predictions, average="micro"
    )
    recall = recall_score(validation_label, svm_validation_predictions, average="micro")
    F1_score = f1_score(validation_label, svm_validation_predictions, average="micro")
    accuracy = accuracy_score(validation_label, svm_validation_predictions)
    print("Precision_Val_SVM",",",  "Recall_Val_SVM", ",", "F1_score_Val_SVM", ",", "Accuracy_Val_SVM")
    print(precision, recall, F1_score, accuracy)

    svm_test_predictions = svm.predict(cv_test_features)
    precision = precision_score(test_label, svm_test_predictions, average="micro")
    recall = recall_score(test_label, svm_test_predictions, average="micro")
    F1_score = f1_score(test_label, svm_test_predictions, average="micro")
    accuracy = accuracy_score(test_label, svm_test_predictions)
    print("Precision_Test_SVM",",",  "Recall_Test_SVM", ",", "F1_score_Test_SVM", ",", "Accuracy_Test_SVM")
    print(precision, recall, F1_score, accuracy)
