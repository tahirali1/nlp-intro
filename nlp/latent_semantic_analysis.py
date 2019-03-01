import  nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class LatentSemanticAnalysis(object):

    def __init__(self):
        self.dataset = ["The amount of pollution is increasing day by day",
                        "The concert was just great",
                        "I love to see Gordon Ramsay cook",
                        "Google is introducing a new technology",
                        "AI Robots are examples of great technology present today",
                        "All of us were signing in the concert",
                        "We have launch campaigns to stop population and global warming"]

    def process_data(self):
        self.dataset = [line.lower() for line in self.dataset]

        vectorizer = TfidfVectorizer()
        x = vectorizer.fit_transform(self.dataset)

        lsa = TruncatedSVD(n_components=4, n_iter=100)
        lsa.fit(x)

        concept_words = {}

        terms = vectorizer.get_feature_names()
        for i, comp in enumerate(lsa.components_):
            compontent_terms = zip(terms, comp)
            sorted_terms = sorted(compontent_terms, key=lambda y: y[1], reverse=True)
            sorted_terms = sorted_terms[:10]
            concept_words["Concept "+str(i)] = sorted_terms

        for key in concept_words.keys():
            sentence_scores = []
            for sentence in self.dataset:
                words = nltk.word_tokenize(sentence)
                score = 0
                for word in words:
                    for word_with_score in concept_words[key]:
                        if word == word_with_score[0]:
                            score += word_with_score[1]
                sentence_scores.append(score)
            print("\n"+key+":")
            for sentence_sc in sentence_scores:
                print(sentence_sc)
