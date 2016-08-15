#!/usr/bin/python3
import pandas as pd
import numpy as np
import string
import nltk

import itertools
from collections import defaultdict
from collections import namedtuple

from math import log


class DataParser(object):

    def __init__(self):
        self._df = None
        self.load()
        self.prepare()

    @property
    def data(self):
        """DataFrame that represents the spam corpus."""
        return self._df

    # TODO: download directly
    def load(self):
        sms_spam_corpus = './sms_spam_corpus_01/english_big.txt'
        self._df = pd.read_csv(sms_spam_corpus,
                               delimiter='|',
                               encoding='latin1')

    def prepare(self):
        self._df['body'] =  self._df['body'].\
            str.replace(r'[0-9]+', '1').\
            str.translate(str.maketrans(string.punctuation,
                          ' ' * len(string.punctuation))).\
            str.replace(r'[^\x00-\x7F]+', ' ').\
            str.lower()


class Separator(object):

    def __init__(self, data, field):
        self._train, self._test = self._make_separation(data=data,
                                                        field=field,
                                                        train_percent=0.8)

    @property
    def test(self):
        """Test part of dataset."""
        return self._train

    @property
    def train(self):
        """Train part of dataset."""
        return self._test

    def _make_separation(self, data, field, train_percent):
        msk = np.random.rand(len(data)) < train_percent
        data[field] = data[field].apply(nltk.word_tokenize)
        data[field] = data[field].apply(np.array)
        return data[msk], data[~msk]


class NaiveBayesModel(object):

    def __init__(self, train):
        self._model = None
        self._fit_model(train_set=train)

    @property
    def trained_model(self):
        """Trained Naive Bayes model."""
        return self._model

    @staticmethod
    def _create_words_map(words, counts):
        words_map = defaultdict(int)
        for i in range(len(words)):
            words_map[words[i]] = counts[i]
        return words_map

    def _fit_model(self, train_set):
        lists_spam_words = train_set[train_set['class'] == 'spam']['body']
        all_spam_words = np.array(list(itertools.chain(*lists_spam_words)))

        lists_ham_words = train_set[train_set['class'] == 'ham']['body']
        all_ham_words = np.array(list(itertools.chain(*lists_ham_words)))

        spam_count_map = self._create_words_map(*np.unique(all_spam_words,
                                                           return_counts=True))
        ham_count_map = self._create_words_map(*np.unique(all_ham_words,
                                                          return_counts=True))

        spam_words = np.unique(all_spam_words)
        ham_words = np.unique(all_ham_words)

        ham = namedtuple('HamModel', ['words', 'count_map'])
        spam = namedtuple('SpamModel', ['words', 'count_map'])
        self._model = ham(ham_words, ham_count_map), spam(spam_words, spam_count_map)


class NaiveBayesClassifier(object):

    def __init__(self, model):
        self.model = model

    def get_words(self, msg):
        prepared_msg = msg.replace(r'[0-9]+', '1').\
            translate(None, string.punctuation.translate(None, '"')).\
            replace(r'[^\x00-\x7F]+', ' ').\
            lower()
        return np.array(nltk.word_tokenize(prepared_msg))

    def classify_all(self, data):
        count = 0
        for row in data.iterrows():
            classified = self.classify(row[1][0])
            real = str(row[1][1])
            if classified == real:
                count += 1
            row[1][2] = self.classify(row[1][0])
        print('result: %s / %s' % (count, len(data)))

    def classify(self, msg):
        ham, spam = self.model
        l_ham = len(ham[0])
        l_spam = len(spam[0])

        p_ham = log(l_ham) / (log(l_ham) + log(l_spam))
        p_spam = log(l_spam) / (log(l_ham) + log(l_spam))

        l_all_spam = len(spam[1])
        l_all_ham = len(ham[1])
        for word in msg:
            if ham.count_map[word]:
                p_ham += log(ham.count_map[word] / l_all_spam)
            if spam.count_map[word]:
                p_spam += log(spam.count_map[word] / l_all_ham)

        result_class = 'ham' if p_ham < p_spam else 'spam'
        #print('spam: %s, ham: %s, for msg: %s, result: %s' % (p_ham, p_spam, str(msg), result_class))
        return result_class

    def test_model(self, test_set):
        test_set.insert(len(test_set.columns), 'classify_class', '')
        self.classify_all(test_set)

if __name__ == "__main__":
    # TODO: make composition
    parser = DataParser()
    separated = Separator(parser.data, 'body')
    trained_model = NaiveBayesModel(separated.train).trained_model
    classifier = NaiveBayesClassifier(trained_model)
    classifier.test_model(separated.test)
    print(separated.test)
