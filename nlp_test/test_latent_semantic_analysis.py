import unittest
from nlp.latent_semantic_analysis import LatentSemanticAnalysis


class TestLatentSemanticAnalysis(unittest.TestCase):

    def setUp(self):
        self.nlpCore = LatentSemanticAnalysis()

    def test_process_data(self):
        self.nlpCore.process_data()
        self.assertEqual(21, 21)


if __name__ == '__main__':
    unittest.main()
