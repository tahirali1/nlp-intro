import unittest
from nlp.negation_word import NegationWord


class TestNGramModel(unittest.TestCase):

    def setUp(self):
        self.line = "I was not happy with the team's performance"
        self.model = NegationWord()

    def test_negative_word_tracking(self):
        self.model.track_negative_word(self.line)
        ret_string = self.model.replace_negative_with_opposite(self.line)
        self.assertEqual("I was unhappy with the team's performance", ret_string)


if __name__ == '__main__':
    unittest.main()
