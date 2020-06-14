import email
import re
import numpy as np
from bs4 import BeautifulSoup
# import pymorphy2
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wordnet

from scipy.io import loadmat
from sklearn.svm import SVC
from joblib import dump, load

import os

lemmatizer = WordNetLemmatizer()

tag_dict = {"J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV}

idx, words = np.loadtxt('vocab.txt', dtype=np.dtype([('idx', np.int), ('word', np.unicode_, 255), ]), unpack=True)
vocabulary = dict(zip(words, idx))

# for russian
# def normalize_word(word):
#     morph_analyzer = pymorphy2.MorphAnalyzer(lang='eng')
#     parse_result = morph_analyzer.parse(word)
#     print(parse_result[0].inflect({'sing', 'nomn'}))
#     return parse_result[0].normal_form
#
#
# while 1:
#     word = input('Введите русское слово:\n')
#     parse_result = MORPH.parse(word)
#     print(parse_result[0].inflect({'sing', 'nomn'}).word, parse_result[0].normal_form)


def normalize_string(_string):

    soup = BeautifulSoup(_string.lower(), features="html.parser")

    __string = soup.get_text()
    __string = re.sub(r'http[s]?://[\w\-_.=\&\?/]+', '_httpref_', __string)
    __string = re.sub(r'[\w\-_]+@[\w\-_.]+', '_emailref_', __string)
    __string = re.sub(r'\$', ' _dollar_ ', __string)
    __string = re.sub(r'[^a-zA-Z0-9_\'\-’]+', ' ', __string)
    __string = re.sub(r'\s\d+\s', ' _number_ ', __string)

    parse_result = ' '.join([lemmatizer.lemmatize(word, tag_dict.get(tag[0], wordnet.NOUN)) for word, tag in pos_tag(__string.split())])

    return parse_result


def string_to_features(_string, vocabulary):
    return list(vocabulary[word] for word in _string.split())


def get_email_body(payload):
    if payload.is_multipart():
        return '\n'.join([get_email_body(nested_payload) for nested_payload in payload.get_payload()])
    else:
        return payload.get_payload()


data = loadmat('spamTrain.mat')
X_train = data['X']
y_train = data['y']

data = loadmat('spamTest.mat')
X_test = data['Xtest']
y_test = data['ytest']

clf = load('spam_assasin_based_spam_filter/spamTrain.clf.linear')

# best_score = 0
# best_C = 0
# best_g = 0
# for C in [0.01, 1, 10]:
#     for g in [0.01, 1, 10]:
#
#         clf = SVC(kernel='linear', C=C, gamma=g, verbose=True)
#         clf.fit(X_train, y_train.ravel())
#         score = clf.score(X_test, y_test)
#         if score > best_score:
#             best_score = score
#             best_C = C
#             best_g = g
#
# print(best_score, best_C, best_g)
# 'poly', 'rbf', 'sigmoid'

# clf = SVC(kernel='linear', C=best_C, gamma=best_g)
# clf.fit(X_train, y_train.ravel())
# dump(clf, 'spamTrain.clf.linear')

print(clf.score(X_test, y_test.ravel()))
top_positive_coefficients = np.argsort(clf.coef_[0])[-15:]

# print(normalize_string(text))

top_words = [f'{words[idx]}: {vocabulary[words[idx]]}' for idx in top_positive_coefficients]
print(', '.join(top_words))

# for root, subdirs, files in os.walk('spam'):
#     for email_file in files:
#         print(f'{root}\\{email_file}\n')
#         email_file_text = open(f'{root}\\{email_file}', 'r', encoding='ansi').read()
#         email_object = email.message_from_string(email_file_text)
#         print(get_email_body(email_object))

# read_files('spam', X, y, is_spam=1)