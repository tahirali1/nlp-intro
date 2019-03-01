import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# nltk.download()


class NlpCore:

    def __init__(self, long_string):
        self.long_str = long_string

    def tokenize_sentences(self):
        return nltk.sent_tokenize(self.long_str)

    def tokenize_word(self):
        return nltk.word_tokenize(self.long_str)

    def stemmer(self, sen):
        stemmer = PorterStemmer()
        for i in range(len(sen)):
            words = nltk.word_tokenize(sen[i])
            words = [stemmer.stem(word) for word in words]
            sen[i] = ' '.join(words)
        return sen

    def lemmatize(self, sen):
        lemmatizer = WordNetLemmatizer()
        for i in range(len(sen)):
            words = nltk.word_tokenize(sen[i])
            words = [lemmatizer.lemmatize(word) for word in words]
            sen[i] = ' '.join(words)
        return sen

    def remove_stop_words(self, sen):
        for i in range(len(sen)):
            words = nltk.word_tokenize(sen[i])
            words = [word for word in words if word not in stopwords.words('english')]
            sen[i] = ' '.join(words)
        return sen

    def parts_of_speech(self):
        words = nltk.word_tokenize(self.long_str)
        tagged_words = nltk.pos_tag(words)
        words_tags = []
        for tw in tagged_words:
            words_tags.append(tw[0] + '_' + tw[1])
        return ' '.join(words_tags)

    def name_entities_recognization(self, data):
        t = nltk.ne_chunk(
            nltk.pos_tag(
                nltk.word_tokenize(data)
            )
        )
        t.draw()

    def name_entities_recognization_sen(self, data):
        t = nltk.ne_chunk(nltk.pos_tag(data))
        t.draw()


