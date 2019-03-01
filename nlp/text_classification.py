import re
import pickle
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('stopwords')


class TextClassfication(object):

    def __init__(self, data_files_path):
        self.data_files_path = data_files_path




    def import_dataset(self):
        reviews = load_files(self.data_files_path)
        x, y = reviews.data, reviews.target

        # store as pickle file
        # persis data set
        with open('x.pickle', 'wb') as f:
            pickle.dump(x, f)

        with open('y.pickle', 'wb') as f:
            pickle.dump(y, f)

        # unpickie the data set
        # don't need as we already loaded the data into memory
        with open('x.pickle', 'rb') as f:
            x = pickle.load(f)

        with open('y.pickle', 'rb') as f:
            y = pickle.load(f)


        corpus = []
        for i in range(0, len(x)):
            review = re.sub(r'\W', ' ', str(x[i]))
            review = review.lower()
            # remove single char word
            review = re.sub(r'\s+[a-z]\s+', ' ', review)
            review = re.sub(r'^/s+', ' ', review)
            corpus.append(review)

        # vector
        vectorizer = TfidfVectorizer(max_features=2000, min_df=3, max_df=0.6, stop_words=stopwords.words('english'))
        X = vectorizer.fit_transform(corpus).toarray()
        # split test and train data
        text_train, text_test, sent_train, sent_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # training the model
        classifier = LogisticRegression()
        classifier.fit(text_train, sent_train)

        sent_predicted = classifier.predict(text_test)

        cm = confusion_matrix(sent_test, sent_predicted)
        result = (cm[0][0] + cm[1][1]) / 4
        print(result)

        # persis our model
        with open('classifier.pickle', 'wb') as f:
            pickle.dump(classifier, f)

        with open('tfidfmodel.pickle', 'wb') as f:
            pickle.dump(vectorizer, f)

        # unpickle the classifier and vectorizer to test on data

        with open('classifier.pickle', 'rb') as f:
            clf = pickle.load(f)

        with open('tfidfmodel.pickle', 'rb') as f:
            tfidf = pickle.load(f)

        print(clf)
        print(tfidf)

        sample = ["you are nice person man, have a good life"]
        sample = tfidf.transform(sample).toarray()
        print(clf.predict(sample))

        neg_sample = ["they does not even response back to the customer's query"]
        neg_sample = tfidf.transform(neg_sample).toarray()
        print(clf.predict(neg_sample))

