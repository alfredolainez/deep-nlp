"""
Script to compute word vectors from the reviews
"""
from gensim.models.word2vec import Word2Vec
from spacy.en import English
from regression import BaseBowRegressor
from language import tokenize_document

# better tokenizer
nlp = English()

NUM_PARTITIONS = 70
WINDOW_SIZE = 4
VECTOR_SIZE = 100
MODEL_FILE = "w2v_%d_parts_%d_vector_%d_window" % (NUM_PARTITIONS, VECTOR_SIZE, WINDOW_SIZE)


reviews_texts, useful_votes, funny_votes, cool_votes, review_stars = BaseBowRegressor.get_reviews_data(range(1, NUM_PARTITIONS))

sentences = [tokenize_document(txt) for txt in enumerate(reviews_texts)]

# build the word2vec model and save it
w2v = Word2Vec(sentences=sentences, size=VECTOR_SIZE, alpha=0.025, window=WINDOW_SIZE, min_count=2, sample=1e-5, workers=4, negative=10)
w2v.init_sims(replace=True)
w2v.save(MODEL_FILE)
