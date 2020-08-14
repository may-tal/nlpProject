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
    The function print ------------------------------------
    :param train_df: the train data
    :return:
    """
    tot = len(train_df)
    # l = train_df["label"]
    # print(l.values)
    print(train_df.label.value_counts())
    print(train_df.label.value_counts() / tot)
    sns.distplot(train_df.label, kde=False)
    plt.show()


def bag_of_words(train_df, test_df):
    """
    convert data to counted vector, that count how many times each word appears
    :param x_train: training data
    :param x_test: test data
    :return: the converted counted vectors and the mapping - training, test, cv
    """
    cv = CountVectorizer(min_df=3, max_df=0.97, ngram_range=(1,3))
    x_train_counts = cv.fit_transform(train_df.text)
    x_test_counts = cv.transform(test_df.text)

    return x_train_counts, x_test_counts


def bow_character_level_n_grams(train_df, test_df):
    """
    convert data to counted vector, that count how many times each word appears
    :param x_train: training data
    :param x_test: test data
    :return: the converted counted vectors and the mapping - training, test, cv
    """
    cv = CountVectorizer(analyzer='char', ngram_range=(1,6), min_df=3, max_df=0.97)
    x_train_counts = cv.fit_transform(train_df.text)
    x_test_counts = cv.transform(test_df.text)

    return x_train_counts, x_test_counts


def tf_idf(x_train_counts, x_test_counts):
    """
    TF-IDF is a numerical statistic that is intended to reflect how important a word
    is to a document in a collection or corpus. This function add the tf-idt weight to each word
    in the bag of words vector in the inputs.
    :param x_train_counts: bag of words for the train data
    :param x_test_counts: bag of words for the test data
    :return: bag of words with tf-idf for each data sets (train and test)
    """
    tf_transformer = TfidfTransformer(use_idf=False).fit(x_train_counts)
    x_train_tf = tf_transformer.transform(x_train_counts)
    x_test_tfidf = tf_transformer.transform(x_test_counts)
    return x_train_tf, x_test_tfidf


def get_bow_tfidf(data):
    train_df, test_df = split_to_train_and_test(data)
    data_exploration(train_df)
    x_train_counts, x_test_counts = bag_of_words(train_df, test_df)
    x_train_tf, x_test_tfidf = tf_idf(x_train_counts, x_test_counts)
    x_train_tf = add_feature_from_sentiment_lexicon(train_df, x_train_tf)
    x_test_tfidf = add_feature_from_sentiment_lexicon(test_df, x_test_tfidf)
    return x_train_tf, x_test_tfidf, train_df, test_df


def _convert_txt_file_to_set(path):
    file = open(path, 'r', encoding='utf-8')
    s = set()
    for line in file:
        s.add(line.replace("\n", ""))
    return s

def add_feature_from_sentiment_lexicon(data, features):
    # neg_words = pd.read_csv('sentiment_lexicon/negative_words_he.txt')
    neg_words = _convert_txt_file_to_set('sentiment_lexicon/negative_words_he.txt')
    pos_words = _convert_txt_file_to_set('sentiment_lexicon/positive_words_he.txt')

    # pos_words = pd.read_csv('sentiment_lexicon/positive_words_he.txt')
    neg_feature = []
    pos_feature = []

    for idx, message in data.iterrows():
        count_neg = 0
        count_pos = 0
        for word in message[0].split():
            if word in neg_words:
                count_neg += 1
            if word in pos_words:
                count_pos += 1

        if len(message[0].split()) == 0:
            mass_len = 1
        else:
            mass_len = len(message[0].split())
        pos_feature.append(count_pos/mass_len)
        neg_feature.append(count_neg/mass_len)

    # data["pos_words"] = pos_feature
    # data["neg_words"] = neg_feature
    print(pos_feature)
    print(neg_feature)
    dense_matrix = features.todense()
    dense_matrix = np.insert(dense_matrix, dense_matrix.shape[1], pos_feature, axis=1)
    dense_matrix = np.insert(dense_matrix, dense_matrix.shape[1], neg_feature, axis=1)
    train_matrix = csr_matrix(dense_matrix)

    return train_matrix


def word2vec_model():
    vectors = np.load('words_vectors.npy')
    with open('words_list.txt', encoding="utf-8") as f:
        words = f.read().splitlines()
    # word2vec_df = pd.DataFrame(list(zip(words, vectors)), columns =['Word', 'Vector'])
    if len(words) != len(vectors):
        print("error")
        raise AssertionError
    word2vec_dict = {}
    for i in range(len(words)):
        word2vec_dict[words[i]] = vectors[i]
    return word2vec_dict


def message_vector(word2vec_model, message):
    # remove out-of-vocabulary words
    message = message.split()
    for word in message:
        if word not in word2vec_model:
            print(word)
    # message = [word2vec_model.Vector[word2vec_model['Word']==word].values for word in message if word in list(word2vec_model.Word)]
    message = [word2vec_model[word] for word in message if word in word2vec_model]
    # message = [word2vec_model.Vector[word2vec_model['Word']==word].values for word in message if word in list(word2vec_model.Word)]
    if len(message) == 0:
        return None
    return list(np.mean(message, axis=0))


def get_word2vec(data):
    train_df, test_df = split_to_train_and_test(data)
    data_exploration(train_df)
    model = word2vec_model()
    train_embedding = np.zeros(shape=[len(train_df), 100])
    test_embedding = np.zeros(shape=[len(test_df), 100])
    count = 0

    for message in train_df.text:  # look up each message in model
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
    for message in test_df.text:  # look up each message in model
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














