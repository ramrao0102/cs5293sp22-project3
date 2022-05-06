# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 17:36:15 2022

@author: ramra
"""

import glob
import io
import os
import pdb
import sys
import re

import nltk

from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk
from nltk import tree2conlltags
from nltk.corpus import treebank

nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

import spacy


# nlp = spacy.load("en_core_web_lg")

tokenize_words = nltk.word_tokenize
filename = "data/*.txt"

files_grabbed = glob.glob(filename)


def find_entity(text):

    list4 = []
    list5 = []
    for sent in sent_tokenize(text):
        words = tokenize_words(sent)
        res = [[words[i], words[i + 1]] for i in range(len(words) - 1)]

        list1 = []
        list2 = []

        for chunk in ne_chunk(pos_tag(word_tokenize(sent))):
            if hasattr(chunk, "label") and chunk.label() == "PERSON":
                for c in chunk.leaves():
                    if "Miramax" not in c:
                        if "Phone" not in c:
                            if "Work" not in c:
                                if "Milestone" not in c:
                                    if "Star" not in c:
                                        if "Starwars" not in c:
                                            if "Sideswipe" not in c:
                                                if "Search" not in c:
                                                    if "Where" not in c:
                                                        if "Shrooms" not in c:
                                                            if "Shimmer" not in c:
                                                                list1.append(c[0])
        for j in range(len(res)):
            for i in range(len(list1) - 1):
                if list1[i] == res[j][0] and list1[i + 1] == res[j][1]:
                    str1 = list1[i] + " " + list1[i + 1]
                    list2.append(str1)

        # print(list1)
        # print(list2)

        list3 = []

        for i in range(len(list1)):
            for j in range(len(list2)):
                if list1[i] in list2[j]:
                    list3.append(list1[i])
                    break

        diff = list(set(list1) - set(list3))

        # print(diff)

        for i in range(len(diff)):
            for j in range(len(list2)):
                if list1.index(str(diff[i])) == 0:
                    list2.insert(0, list1[list1.index(str(diff[i]))])

                elif list1[list1.index(str(diff[i])) - 1] in str(list2[j]):
                    list2.insert(j + 1, list1[list1.index(str(diff[i]))])

        if len(list2) == 0:
            list2.extend(diff)

        if len(list2) >= 1 and len(list2) <= 2:

            list4.append(list2)
            list5.append(sent)

    # print(list4)
    return list4, list5


def redact_name(list1, text):

    data1 = text

    list2 = list1

    for i in range(len(list2)):

        if i == 0:

            if list2[i].isspace():

                str1 = re.split(r'\s+', list2[i])

                data1 = data1.replace(str1[0], "\u2588" * len(str1[0]))

                data1 = data1.replace(str1[0], "\u2588" * len(str1[0]))

            else:

                data1 = data1.replace(list2[i], "\u2588" * len(list2[i]))

    return data1


def writepath():

    filename = "output12.tsv"

    return filename


if __name__ == "__main__":

    filename = writepath()

    myfile = open(filename, "w", encoding="utf-8")

    j = 0

    for i in files_grabbed:
        #print(i)
        file_name = i
        with open(i) as f:
            text = f.read()

        text = text.replace(".<br /><br", ".")
        text = text.replace("<br", "")
        text = text.replace(r'\s{2,}', " ")

        list4, list5 = find_entity(text)
        # print(list4)

        # print(list4)

        if len(list4) > 0:
            # print(list4)
            for i in range(len(list5)):
                # print(list4[i][0] , list5[i])
                redacted_text = redact_name(list4[i], list5[i])
                redacted_text = redacted_text.replace("/>", " ")
                redacted_text = redacted_text.replace("!", "")
                redacted_text = redacted_text.replace("'s", " ")

                #print(list4[i][0], redacted_text)
                if redacted_text != "":
                    j += 1
                    #print(j)
                    if j <= 50:
                        sent_type = "training"

                    if j > 50 and j <= 80:
                        sent_type = "validation"

                    if j > 80 and j <= 90:
                        sent_type = "testing"

                    if j < 91:

                        writelist = []
                        if len(list4[i]) == 1:
                            # print(list4[i][0])
                            writelist.append("RamRao")
                            writelist.append(sent_type)
                            writelist.append(str(list4[i][0]))
                            writelist.append(redacted_text)

                            myfile.write("\t".join(writelist) + "\n")

                        if len(list4[i]) == 2:
                            # print(list4[i][0], list4[i][1])
                            writelist.append("RamRao")
                            writelist.append(sent_type)
                            writelist.append(str(list4[i][0]))
                            writelist.append(redacted_text)

                            myfile.write("\t".join(writelist) + "\n")
