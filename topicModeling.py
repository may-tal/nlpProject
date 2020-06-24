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


def get_topics(vh, words_vocab, num_top_words=8):
    """
    this function return the top words for each topic
    """
    top_words = lambda t: [words_vocab[i] for i in np.argsort(t)[:-num_top_words-1:-1]]
    topic_words = ([top_words(t) for t in vh])
    return [' '.join(t) for t in topic_words]


def print_messages_by_topic(data, u, num_of_massages=8):
    """
    this function print each topic and the messages that match to this topic
    """
    for i in range(6):
        print("------------------------")
        print("topic number " + str(i))
        cur_column = u[:, i]
        topic_messages = [i for i in np.argsort(cur_column)[:-num_of_massages-1:-1]]
        for j in topic_messages:
            print("message ID " + str(j))
            print(data.text.iloc[j])
            print("label: " + str(data.label.iloc[j]))

def svd(vectors):
    """
    This function return the singular value decomposition
    u- from message to topic
    s- perfects the issues by importance
    vh- from topic to words
    """
    u, s, vh = linalg.svd(vectors, full_matrices=False)
    return u, s, vh


def nmf(vectors, num_of_topics=8):
    """
    return nmf decomposition
    w1- connects topics to documents
    h1- connects topics to terms
    """
    clf = decomposition.NMF(n_components=num_of_topics, random_state=1)
    w1 = clf.fit_transform(vectors)
    h1 = clf.components_
    return w1, h1

def lda()


