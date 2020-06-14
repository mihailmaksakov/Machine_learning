import numpy as np
import re

from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wordnet

from bs4 import BeautifulSoup

import os
import email

lemmatizer = WordNetLemmatizer()

tag_dict = {"J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV}


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

    result = np.zeros((1, len(vocabulary.values())))

    for word in _string.split():
        if word in vocabulary:
            result[0][vocabulary[word]] = 1

    return result


def get_email_body(payload):
    if payload.is_multipart():
        return '\n'.join([get_email_body(nested_payload) for nested_payload in payload.get_payload()])
    else:
        return payload.get_payload()


def get_email_body_iterator(dir):
    for root, subdirs, files in os.walk(dir):
        for email_file in files:
            email_file_text = open(f'{root}\\{email_file}', 'r', encoding='ansi').read()
            email_object = email.message_from_string(email_file_text)
            yield get_email_body(email_object)


def create_vocabulary(debug=False):

    result = dict()
    word_count = dict()

    j = 1
    for email_body in get_email_body_iterator('spam_assasin_based_spam_filter/spam'):
        if debug:
            j = j + 1
            if j > 100:
                 break
        for n_word in normalize_string(email_body).split():
            if n_word in word_count:
                word_count[n_word] += 1
            else:
                word_count[n_word] = 1

    j = 1
    for email_body in get_email_body_iterator('spam_assasin_based_spam_filter/nonspam'):
        if debug:
            j = j + 1
            if j > 100:
                 break
        for n_word in normalize_string(email_body).split():
            if n_word in word_count:
                word_count[n_word] += 1
            else:
                word_count[n_word] = 1


    idx = 0

    for word, count in word_count.items():
        if count > 100:
            result[word] = idx
            idx += 1

    return result


def read_files(directory, spam, vocabulary):

    features_matrix = np.zeros((0, len(vocabulary.values())))
    classification_array = np.zeros((0, 1))

    for root, subdirs, files in os.walk(directory):
        for email_file in files:
            email_file_text = open(f'{root}\\{email_file}', 'r', encoding='ansi').read()
            email_object = email.message_from_string(email_file_text)
            email_body_norm = normalize_string(get_email_body(email_object))
            features_matrix = np.vstack((features_matrix, string_to_features(email_body_norm, vocabulary)))
            classification_array = np.vstack((classification_array, np.array([spam])))
            print(f'{root}\\{email_file}')

    return features_matrix, classification_array