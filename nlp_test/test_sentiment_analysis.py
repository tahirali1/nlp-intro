import unittest
import re

from nlp.sentiment_analysis import TwitterSentimentAnalysis


class TestTwitterSentimentAnalysis(unittest.TestCase):

    def test_model(self):

        twitter = TwitterSentimentAnalysis()
        twitter.do_work()


if __name__ == '__main__':
    unittest.main()
