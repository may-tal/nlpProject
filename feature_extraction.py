import random
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix


def split_to_train_and_test(data_df, random_state=None):
    """
    prepare the data to classifying format
    :param data_df: the data to classify
    :param random_state: a seed to split the data by (keep empty if you want a random seed)
    :return: x_train, x_test, y_train, y_test
    """
    if random_state is None:
        random_state = random.randint(0, 1000)

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(data_df['text'], data_df['label'], random_state=random_state)
    train_df = pd.DataFrame({"text": x_train, "label": y_train})
    test_df = pd.DataFrame({"text": x_test, "label": y_test})
    return train_df, test_df


def data_exploration(train_df):
    """
    The function print the label distribution of the training data - the number pf samples with label
    1 and with label 0
    :param train_df: the train data
    """
    tot = len(train_df)
    print(train_df.label.value_counts())
    print(train_df.label.value_counts() / tot)
    sns.distplot(train_df.label, kde=False)
    plt.show()


def bag_of_words(train_df, test_df):
    """
    convert data to counted vector, that count how many times word with min_df=3 and max_df=97% appears.
    binary features indicating the presence of word unigrams, bigrams and trigrams.

    :param x_train: training data
    :param x_test: test data
    :return: the converted counted vectors and the mapping - training, test, cv
    """
    cv = CountVectorizer(min_df=3, max_df=0.97, ngram_range=(1,3))
    x_train_counts = cv.fit_transform(train_df.text.astype(np.str))
    x_test_counts = cv.transform(test_df.text.astype(np.str))

    return x_train_counts, x_test_counts


def bow_character_level_n_grams(train_df, test_df):
    """
     binary features indicating the presence of character n-gram
    (without crossing word boundaries). Character n-grams provide some
     abstraction from the word level and provide robustness to the spelling variation that
     characterises social media data.
    :param x_train: training data
    :param x_test: test data
    :return: the converted counted vectors and the mapping - training, test, cv
    """
    cv = CountVectorizer(analyzer='char_wb', ngram_range=(1,6), min_df=3, max_df=0.97)
    x_train_counts = cv.fit_transform(train_df.text.astype(np.str))
    x_test_counts = cv.transform(test_df.text.astype(np.str))

    return x_train_counts, x_test_counts


def tf_idf(x_train_counts, x_test_counts):
    """
    The TF-IDF (term frequency-inverse document frequency) is a measure of the importance of a word in a document
    within a collection of documents, thereby taking into account the frequency of occurrence of a word in the entire
    corpus as a whole and within each document. G This function add the tf-idt weight to each word
    in the bag of words vector in the inputs.
    :param x_train_counts: bag of words for the train data
    :param x_test_counts: bag of words for the test data
    :return: bag of words with tf-idf for each data sets (train and test)
    """
    tf_transformer = TfidfTransformer(use_idf=False).fit(x_train_counts)
    x_train_tf = tf_transformer.transform(x_train_counts)
    x_test_tfidf = tf_transformer.transform(x_test_counts)
    return x_train_tf, x_test_tfidf


def get_bow_tfidf(data, flag):
    """
    create bow+tf-idf features
    :param data: The data for creating the features
    :param flag: if True the features are bow_tf-idf and if False the features are  bow_character_n_grams + tf-idf
    :return:
    """
    train_df, test_df = split_to_train_and_test(data)
    data_exploration(train_df)
    if flag:  # bag of words with sentiment lexicon
        x_train_counts, x_test_counts = bag_of_words(train_df, test_df)
    else:  # bow character level n grams
        x_train_counts, x_test_counts = bow_character_level_n_grams(train_df, test_df)
    x_train_tf, x_test_tfidf = tf_idf(x_train_counts, x_test_counts)
    if flag: # bag of words with sentiment lexicon
        x_train_tf = add_feature_from_sentiment_lexicon(train_df, x_train_tf)
        x_test_tfidf = add_feature_from_sentiment_lexicon(test_df, x_test_tfidf)
    return x_train_tf, x_test_tfidf, train_df, test_df


def _convert_txt_file_to_set(path):
    """
    convert txt file to set for more efficient search
    :param path: path for the file's directory
    :return: txt file as set
    """
    file = open(path, 'r', encoding='utf-8')
    s = set()
    for line in file:
        s.add(line.replace("\n", ""))
    return s

def add_feature_from_sentiment_lexicon(data, features):
    """
    create sentiment features(for positive words and negative words) from sentiment lexicon.
    :param data: The data for creating the features
    :param features: set of feature to add them sentiments features
    :return: train matrix oa all the features
    """
    neg_words = _convert_txt_file_to_set('sentiment_lexicon/negative_words_he.txt')
    pos_words = _convert_txt_file_to_set('sentiment_lexicon/positive_words_he.txt')

    neg_feature = []
    pos_feature = []

    # the feature is calculate by number to sentimants words ic the sentence divide by number of words in the sentence
    for idx, message in data.iterrows():
        count_neg = 0
        count_pos = 0
        for word in str(message[0]).split():
            if word in neg_words:
                count_neg += 1
            if word in pos_words:
                count_pos += 1
        # if there is no sentiment feature in the sentence the score is 0
        if len(str(message[0]).split()) == 0:
            mass_len = 1
        else:
            mass_len = len(str(message[0]).split())
        pos_feature.append(count_pos/mass_len)
        neg_feature.append(count_neg/mass_len)

    # add the semantic features to the general matrix features
    dense_matrix = features.todense()
    dense_matrix = np.insert(dense_matrix, dense_matrix.shape[1], pos_feature, axis=1)
    dense_matrix = np.insert(dense_matrix, dense_matrix.shape[1], neg_feature, axis=1)
    train_matrix = csr_matrix(dense_matrix)

    return train_matrix


def word2vec_model():
    """
    loading pre-trained word2vec model as a dictionary
    :return: word2vec dictionary while the keys is the words tnw the values is the embedding vector
    """
    vectors = np.load('words_vectors.npy')
    with open('words_list.txt', encoding="utf-8") as f:
        words = f.read().splitlines()
    if len(words) != len(vectors):
        print("error")
        raise AssertionError
    word2vec_dict = {}
    for i in range(len(words)):
        word2vec_dict[words[i]] = vectors[i]
    return word2vec_dict


def message_vector(word2vec_model, message):
    """
    create word2vec for message - for each word in the message
    take the average of all the word vectors in a sentence
    and it will represent the message vector.
    :param word2vec_model: dictionary of pre-trained word2vec model
    :param message: the message for creating the features
    :return: message embedding vector
    """
    # remove out-of-vocabulary words
    message = message.split()
    # for word in message:
    #     if word not in word2vec_model:
    #         print(word)
    message = [word2vec_model[word] for word in message if word in word2vec_model]
    if len(message) == 0:
        return None
    return list(np.mean(message, axis=0))


def get_word2vec(data):
    """
    Build word2vec + TF-IDF embedding for all the data
    :param data: The data for creating the features
    :return: x_train_tf, x_test_tfidf, train_df, test_df
    """
    train_df, test_df = split_to_train_and_test(data)
    data_exploration(train_df)
    model = word2vec_model()
    train_embedding = np.zeros(shape=[len(train_df), 100])
    test_embedding = np.zeros(shape=[len(test_df), 100])
    count = 0

    for message in train_df.text.astype(np.str):  # look up each message in model
        embedded_message = message_vector(model, message)
        if embedded_message is None:
            count += 1
            # print("---------train-----------")
            # # print(message)
            # print("--------------------")
            continue
        train_embedding[count] = embedded_message
        count += 1

    count = 0
    for message in test_df.text.astype(np.str):  # look up each message in model
        embedded_message = message_vector(model, message)
        if embedded_message is None:
            count += 1
            # print("---------test-----------")
            # # print(message)
            # print("--------------------")
            continue
        test_embedding[count] = embedded_message
        count += 1

    x_train_tf, x_test_tfidf = tf_idf(train_embedding, test_embedding)
    return x_train_tf, x_test_tfidf, train_df, test_df
