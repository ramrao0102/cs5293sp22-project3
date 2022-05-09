# This is Ramkishore Rao's Project3 Readme File

## Introduction

This project consists of taking text files from the IMDB database and generating a file that contains redacted <br>
sentence that consist of text surrounding the redacted word.  The redacted word is the name in the sentence. <br>
The focus of the second part of the project is to find the redacted word with the use of classfication models. <br>

## RAM Requirement

Use, atleast 8 GB RAM, for the VM instance for execution of code.<br>

## How to Run the Code

1) Clone the github repostory. <br>
2) Run pipenv install pytest from command line. <br>
3) Run pipenv run python -m pytest -v from command line. <br>
4) Run pipenv run python project3.py from command line.

## Packages Used 

1) re, urllib, sklearn, NLTK, pandas, numpy. SPACY is included but not utilized. <br>
2) these packages do not need to be reinstalled as they will be automatically loaded when pipenv instal pytest is run from the command line.

## Redaction Process

In order to complete the redaction, the following process is followed: <br>
    
1) The code to run from command line is pipenv run python redactor.py. <br> 
2) The .txt files from the IMDB database are placed in the /data folder under the root. <br>
3) Funtions have been created to find the entity_name == "PERSON" in each sentence and then to redact the name. <br>
   NLTK's ne_chunk is used to perform this activity Functions are find_entity and redact_name. <br>
4) The code then generates 90 sentences with the redacted_name and the redacted_sentence as tab separated file. <br>
5) The tab separated file has 50 training sentences, 30 validation sentences, and 10 test sentences. <br>
6) This file is then combined with the unredactor.tsv fetched from cegme repository and creates a final.tsv file which is then used to upload to cegme.<br>

## Unredaction Process Preliminary Work

For the unredaction, the following process is utilized: <br>
1) The code to run from the command line is pipenv run python project3.py. <br>
2) "unredactor.tsv" file is downloaded from "cegme" repository using urllib package with download_file function. <br>
3) The file is then read into a pandas dataframe with create_dataframe function. <br>
4) NA lines are removed form the pandas dataframe. <br>
5) Column positions are swapped to move the "Person" column to the far right. This is the true label for the classification with swap_columns function.<br>
6) Text is normalized to remove select features that do not contribute to the classification with normalize_text function. <br>
7) The redaction characters are removed from the features. <br>
8) First or Last Names are consolidate when First and Last Names are both present in string with consolidate_names function. <br>
9) Additional column is created in the data_frame with features (16 words max) that surround the redacted_name with feature_selection function. <br>
10) After this step, the dataframe is broken out into training, test, and validation dataframes. <br>

## Unredaction Classification

For the classification, i.e., to find the missing or the redacted_name following process is used:

1) The features, i.e., the context around the redacted_name are vectorized using count vectorizer and TfIDf vectorizers<br>
in the train, validation, and test datasets. <br>
2) Three models are used to train the data, Multinomial Bayes, Logistic Regression, and Support Vector Machines.<br>
3) Multinomial Bayes has been commented out in the final version. <br>
4) The model is trained on the training dataset and then checked on the validation and the testing datasets. <br>
5) Precision, Recall, F1_score, and Accuracy results for each model for both the validation and the test datasets are <br>
printed to the console. <br>



## Testing of the Code

The developed program is tested using pytest. <br>

1) pytest can be initiated by running the command pipenv run python -m pytest -v. <br>
2) 4 test methods have been developed. <br>
3) test_download.py checks to make sure that the unredactor.tsv is downloaded from cegme repository. <br>
4) test_dataframe.py checks to make sure that the pandas dataframe is created from the unredactor file. <br>
5) test_istheredata.py checks the redactor.py process and makes sure that the 90 sentences have been created following redaction. <br>
6) test_prediction checks to make sure that for one of the models, Multinominal Bayes, the precision, recall, F1_score, and accuracy are greater than 0. <br>

## Video Link

It is inclded here as: Project305072022.mp4
Video Link to Video Hosted on Vimeo Site is provided below.

https://vimeo.com/707420262
