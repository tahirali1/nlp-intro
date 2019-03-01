import nltk
import re
import heapq
import numpy as np


class BowModel:

    def __init__(self, long_string):
        self.long_str = long_string

    def __prepare_data(self):
        dataset = nltk.sent_tokenize(self.long_str)
        for i in range(len(dataset)):
            dataset[i] = dataset[i].lower()
            dataset[i] = re.sub(r'\W', ' ', dataset[i])
            dataset[i] = re.sub(r'\s+', ' ', dataset[i])
        return dataset

    def __create_histogram(self, dataset):
        word_count = {}
        for data in dataset:
            words = nltk.word_tokenize(data)
            for word in words:
                if word not in word_count.keys():
                    word_count[word] = 1
                else:
                    word_count[word] += 1
        return word_count

    def __frequent_words(self, word_count):
        freq_words = heapq.nlargest(100, word_count, key=word_count.get)
        return freq_words

    def build_bow_model(self):
        dataset = self.__prepare_data()
        word_count = self.__create_histogram(dataset)
        freq_words = self.__frequent_words(word_count)

        x = []
        for sentence in dataset:
            vector = []
            for word in freq_words:
                if word in nltk.word_tokenize(sentence):
                    vector.append(1)
                else:
                    vector.append(0)
            x.append(vector)
        x = np.array(x)
        return x
