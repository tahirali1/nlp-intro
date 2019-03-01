import unittest
import re
from nlp.text_classifier import TextClassifier


class TestTextClassification(unittest.TestCase):

    def setUp(self):
        self.model = TextClassifier('C:\\Users\\taali\\Downloads\\dataset')

    def test_model(self):
        reviews_data, reviews_target = self.model.import_dataset()

        self.assertGreater(len(reviews_target), 0)

        process_data = self.model.pre_process_data(reviews_data)

        self.assertTrue(re.match(r"[a-zA-Z0-9 ]*$", process_data[0]))

        result = self.model.build_model(process_data, reviews_target)

        self.assertGreater(result, 80)

        # test on sample data
        clf, tfidf = self.model.get_classifier_cectorizer()
        sample = ["you are nice person man, have a good life"]
        sample = tfidf.transform(sample).toarray()
        prediction = clf.predict(sample)
        self.assertEqual(prediction, 1)

        neg_sample = ["they does not even response back to the customer's query"]
        neg_sample = tfidf.transform(neg_sample).toarray()
        prediction = clf.predict(neg_sample)
        self.assertEqual(prediction, 0)


if __name__ == '__main__':
    unittest.main()
