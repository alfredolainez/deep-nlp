from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from regression import BaseBowRegressor

import nltk

reviews_texts, _, _, _, _ = BaseBowRegressor.get_reviews_data(range(1,70))
sentences = []
print "Tokenizing sentences..."
for i, review in enumerate(reviews_texts):
    tokens = nltk.word_tokenize(review)
    tokens = [token.lower() for token in tokens]
    sentences.append(LabeledSentence(words=tokens, labels=["REVIEW_" + str(i)]))

print "Doc2Vec"
model = Doc2Vec(sentences, size=100, window=8, min_count=5, workers=4)

