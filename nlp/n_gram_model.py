import nltk


class NGramCharModel(object):

    def __init__(self, text, n_of_ngram):
        self.text = text
        self.gram_length = n_of_ngram

    def build_n_grams(self):

        ngrams = {}
        for i in range(len(self.text) - self.gram_length):
            gram = self.text[i: i + self.gram_length]
            if gram not in ngrams.keys():
                ngrams[gram] = []
            ngrams[gram].append(self.text[i+self.gram_length])
        return ngrams


class NGramWordModel(object):
    def __init__(self, text, n_of_ngram):
        self.text = text
        self.gram_length = n_of_ngram

    def build_word_n_grams(self):
        ngrams = {}
        words = nltk.word_tokenize(self.text)
        for i in range(len(words) - self.gram_length):
            gram = ' '.join(words[i:i+self.gram_length])
            if gram not in ngrams.keys():
                ngrams[gram] = []
            ngrams[gram].append(words[i+self.gram_length])
        return ngrams

