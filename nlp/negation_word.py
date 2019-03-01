import nltk
import re
from nltk.corpus import wordnet


class NegationWord(object):

    def __init__(self):
        pass

    def track_negative_word(self, line):
        words = nltk.word_tokenize(line)
        modified_words = []
        temp_word = ''
        for word in words:
            if word == "not":
                temp_word = "not_"
            elif temp_word == "not_":
                word = temp_word + word
                temp_word = ''
            if word != "not":
                modified_words.append(word)
        return ' '.join(modified_words)

    def replace_negative_with_opposite(self, line):
        words = nltk.word_tokenize(line)
        modified_words = []
        temp_word = ''
        for word in words:
            antonyms = []
            if word == "not":
                temp_word = "not_"
            elif temp_word == "not_":
                for syn in wordnet.synsets(word):
                    for s in syn.lemmas():
                        for a in s.antonyms():
                            antonyms.append(a.name())
                if len(antonyms) >= 0:
                    word = antonyms[0]
                else:
                    word = temp_word + word
                temp_word = ''
            if word != "not":
                modified_words.append(word)
        new_line = ' '.join(modified_words)
        return re.sub("\s+[',]", "'", new_line)

