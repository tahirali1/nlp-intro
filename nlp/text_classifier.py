import re
import pickle
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('stopwords')


class TextClassifier(object):

    def __init__(self, data_files_path):
        self.data_files_path = data_files_path

    def import_dataset(self):
        reviews = load_files(self.data_files_path)
        return reviews.data, reviews.target

    # pre-process the data-set
    def pre_process_data(self, data):
        """ clean the data.
            1- non word char
            2- convert to lower case
            3- remove single word letter
            4- remove single word letter even it is a first word
        """
        corpus = []
        for i in range(0, len(data)):
            review = re.sub(r'\W+', ' ', str(data[i]))
            review = review.lower()
            # remove single char word
            review = re.sub(r'\s+[a-z]\s+', ' ', review)
            review = re.sub(r'^/s+', ' ', review)
            corpus.append(review)
        return corpus

    def build_model(self, training_data, target):

        vectorizer = TfidfVectorizer(max_features=2000, min_df=3, max_df=0.6, stop_words=stopwords.words('english'))
        X = vectorizer.fit_transform(training_data).toarray()
        # split test and train data
        text_train, text_test, sent_train, sent_test = train_test_split(X, target, test_size=0.2, random_state=0)

        # training the model
        classifier = LogisticRegression()
        classifier.fit(text_train, sent_train)

        sent_predicted = classifier.predict(text_test)

        cm = confusion_matrix(sent_test, sent_predicted)
        result = (cm[0][0] + cm[1][1]) / 4

        # persis our model
        with open('classifier.pickle', 'wb') as f:
            pickle.dump(classifier, f)

        with open('tfidfmodel.pickle', 'wb') as f:
            pickle.dump(vectorizer, f)

        return result

    def get_classifier_cectorizer(self):
        with open('classifier.pickle', 'rb') as f:
            clf = pickle.load(f)

        with open('tfidfmodel.pickle', 'rb') as f:
            tfidf = pickle.load(f)

        return clf, tfidf



