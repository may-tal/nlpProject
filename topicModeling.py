from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from sklearn import decomposition
from scipy import linalg
import matplotlib.pyplot as plt


def data_processing(data):
    """
    this function extract all the word counts and return the vectors and the words mapping
    """
    cv = CountVectorizer()
    vectors = cv.fit_transform(data.text).todense()  # (documents, vocab)
    words_vocab = np.array(cv.get_feature_names())  # mapping numbers to words
    return vectors, words_vocab


def svd(vectors):
    """
    This function return the singular value decomposition
    """
    u, s, vh = linalg.svd(vectors, full_matrices=False)
    return u, s, vh


def show_topics(vh, words_vocab, num_top_words = 8):
    top_words = lambda t: [words_vocab[i] for i in np.argsort(t)[:-num_top_words-1:-1]]
    topic_words = ([top_words(t) for t in vh])
    return [' '.join(t) for t in topic_words]
