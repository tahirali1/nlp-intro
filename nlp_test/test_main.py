import unittest
from nlp.main import NlpCore
from nlp.bow_model import BowModel


class TestNlpCore(unittest.TestCase):

    def setUp(self):
        long_str = """Thank you all so very much. Thank you to the Academy. 
                        Thank you to all of you in this room. I have to congratulate the other incredible nominees this year.
                        The Revenant was the product of the tireless efforts of an unbelievable cast and crew.
                        First off, to my brother in this endeavor, Mr. Tom Hardy.
                        Tom, your talent on screen can only be surpassed by your friendship off screen
                        … thank you for creating a transcendent cinematic experience. Thank you to everybody 
                        at Fox and New Regency
                        … my entire team. I have to thank everyone from the very onset of my career … To my parents;
                        none of this would be possible without you. And to my friends, I love you dearly; you know who you are.
                        And lastly, I just want to say this: Making The Revenant was about man's relationship to the natural 
                        world. A world that we collectively felt in 2015 as the hottest year in recorded history.
                        Our production needed to move to the southern tip of this planet just to be able to find snow.
                        Climate change is real, it is happening right now. 
                        It is the most urgent threat facing our entire species,
                        and we need to work collectively together and stop procrastinating.
                        We need to support leaders around the world who do not speak for the big polluters,
                        but who speak for all of humanity, for the indigenous people of the world, 
                        for the billions and billions of underprivileged people out there who would be most affected by this.
                        For our children’s children, and for those people out there whose voices have been drowned
                        out by the politics of greed. 
                        I thank you all for this amazing award tonight. Let us not take this planet for granted.
                        I do not take tonight for granted. Thank you so very much."""
        self.nlpCore = NlpCore(long_str)
        self.sentences = self.nlpCore.tokenize_sentences()

    def test_tokenize_sentences(self):
        self.assertEqual(21, len(self.nlpCore.tokenize_sentences()))

    def test_tokenize_word(self):
        self.assertEqual(346, len(self.nlpCore.tokenize_word()))

    def test_stemmer(self):
        self.assertEqual("thank you all so veri much .", self.nlpCore.stemmer(self.sentences)[0])

    def test_lemmitization(self):
        self.assertEqual("Thank you all so very much .", self.nlpCore.lemmatize(self.sentences)[0])

    def test_remove_stop_words(self):
        self.assertEqual("Thank much .", self.nlpCore.remove_stop_words(self.sentences)[0])

    def test_parts_of_speech(self):
        assert (len(self.nlpCore.parts_of_speech()) > 0)
        bow = BowModel()
        bow.build_bow_model()


if __name__ == '__main__':
    unittest.main()
