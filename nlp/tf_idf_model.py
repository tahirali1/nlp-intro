import nltk
import re
import heapq
import numpy as np


class TfidfFModel:

    def __init__(self, long_string):
        self.long_str = long_string

    def prepare_data(self):
        dataset = nltk.sent_tokenize(self.long_str)
        for i in range(len(dataset)):
            dataset[i] = dataset[i].lower()
            dataset[i] = re.sub(r'\W', ' ', dataset[i])
            dataset[i] = re.sub(r'\s+', ' ', dataset[i])
        return dataset

    def create_histogram(self, dataset):
        word_count = {}
        for data in dataset:
            words = nltk.word_tokenize(data)
            for word in words:
                if word not in word_count.keys():
                    word_count[word] = 1
                else:
                    word_count[word] += 1
        return word_count

    def frequent_words(self, word_count):
        freq_words = heapq.nlargest(100, word_count, key=word_count.get)
        return freq_words

    def prepare_idf_matrix(self, dataset, freq_words):
        word_idfs = {}
        for word in freq_words:
            doc_count = 0
            for data in dataset:
                if word in nltk.word_tokenize(data):
                    doc_count += 1
            word_idfs[word] = np.log((len(dataset)/doc_count)+1)
        print(word_idfs)
        return word_idfs

    def prepare_tf_matrix(self, dataset, freq_words):
        tf_matrix = {}
        for word in freq_words:
            doc_tf = []
            for data in dataset:
                frequncy = 0
                for w in nltk.word_tokenize(data):
                    if word == w:
                        frequncy += 1
                tf_word = frequncy / len(nltk.word_tokenize(data))
                doc_tf.append(tf_word)
            tf_matrix[word] = doc_tf
        print(tf_matrix)
        return tf_matrix

    def td_idf_calculation(self, idf_matrix, tf_marix):
        tfidf_matrix = []
        for word in tf_marix:
            tfidf = []
            for value in tf_marix[word]:
                score = value * idf_matrix[word]
                tfidf.append(score)
            tfidf_matrix.append(tfidf)
        return tfidf_matrix

    def build_tdidf_model(self, score_matrix):
        X = np.array(score_matrix)
        X = np.transpose(X)
        return X

