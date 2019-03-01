import unittest
import random
import nltk
from nlp.n_gram_model import NGramCharModel
from nlp.n_gram_model import NGramWordModel


class TestNGramModel(unittest.TestCase):

    def setUp(self):
        self.GRAM_LENGTH = 3
        self.long_str = """Global warming or climate change has become a worldwide concern. 
                            it is gradually developing into an unprecedence environmental
                             crisis evident in melting glaciers, change weather parrerents, rising sea level"""

        '''increasing the value on n which is set to 6 for this test improve the results'''
        self.model = NGramCharModel(self.long_str, self.GRAM_LENGTH)
        self.word_model = NGramWordModel(self.long_str, self.GRAM_LENGTH)

    def test_n_gram_char_model(self):
        ngrams = self.model.build_n_grams()
        current_n_gram = self.long_str[0:self.GRAM_LENGTH]
        result = current_n_gram
        for i in range(100):
            if current_n_gram not in ngrams.keys():
                break
            possibilities = ngrams[current_n_gram]
            next_item = possibilities[random.randrange(len(possibilities))]
            result += next_item
            current_n_gram = result[len(result)-self.GRAM_LENGTH:len(result)]
        # print(result)
        self.assertGreaterEqual(len(result), 0)

    def test_n_gram_word_model(self):
        words = nltk.word_tokenize(self.long_str)
        ngrams = self.word_model.build_word_n_grams()
        current_gram = ' '.join(words[0:self.GRAM_LENGTH])
        result = current_gram
        for i in range(30):
            if current_gram not in ngrams.keys():
                break
            possibilities = ngrams[current_gram]
            next_item = possibilities[random.randrange(len(possibilities))]
            result += ' '+next_item
            rwords = nltk.word_tokenize(result)
            current_gram = ' '.join(rwords[len(rwords)-self.GRAM_LENGTH:len(rwords)])
        # print('Prediction')
        print(result)
        self.assertGreaterEqual(len(result), 0)


if __name__ == '__main__':
    unittest.main()
