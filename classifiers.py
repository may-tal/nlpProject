import random
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from evaluation import Evaluation


def prepare_data_for_classify(data_df, random_state=None):
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
    cv = CountVectorizer()
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
    :return: bag os words with tf-idf for each data sets (train and test)
    """
    tf_transformer = TfidfTransformer(use_idf=False).fit(x_train_counts)
    x_train_tf = tf_transformer.transform(x_train_counts)
    x_test_tfidf = tf_transformer.transform(x_test_counts)

    return x_train_tf, x_test_tfidf


def majority_classifier(test_df):
    return len(test_df) * [0]


def multinomial_naive_bayes_classifier(x_train_tf, train_df, x_test_tfidf):
    """
    classify data by naive bayes classifier
    :param x_train_tf: training data represented as counted vector
    :param train_df: the training data
    :param x_test_tfidf: test data represented as counted vector
    :return: predicted labels
    """
    #evaluator = Evaluator()
    naive_bayes = MultinomialNB()
    naive_bayes.fit(x_train_tf, train_df.label)
    predictions = naive_bayes.predict(x_test_tfidf)

    #evaluator.evaluate(predictions)
    return predictions

def regression_logistic_classifier(x_train_tf, train_df, x_test_tfidf):
    """
    classify data by regression logistic classifier
    :param x_train_tf: training data represented as counted vector
    :param train_df: the training data
    :param x_test_tfidf: test data represented as counted vector
    :return: predicted labels
    """
    #evaluator = Evaluator()
    regression_logistic = LogisticRegression()
    regression_logistic.fit(x_train_tf, train_df.label)
    predictions = regression_logistic.predict(x_test_tfidf)

    #evaluator.evaluate(predictions)
    return predictions


def get_strongest_words(label, clf):
    """
    get the words that most influenced the classifier
    :param label: violence or not
    :param clf: classifier
    """
    cv = CountVectorizer()
    inverse_dict = {cv.vocabulary_[w]: w for w in cv.vocabulary_.keys()}
    cur_coef = clf.coef_[label]
    word_df=pd.DataFrame({"val":cur_coef}).reset_index().sort_values(["val"],ascending=[False])
    word_df.loc[:, "word"]=word_df["index"].apply(lambda v:inverse_dict[v])
    print(word_df.head(10))


def random_forest_classifier(x_train_tf, train_df, x_test_tfidf):
    """
    classify data by random forest classifier
    :param x_train_tf: training data represented as counted vector
    :param train_df: the training data
    :param x_test_tfidf: test data represented as counted vector
    :return: predicted labels
    """
    #evaluator = Evaluator()
    random_forest = RandomForestClassifier()
    random_forest.fit(x_train_tf, train_df.label)
    predictions = random_forest.predict(x_test_tfidf)

    #evaluator.evaluate(predictions)
    return predictions


def cat_boost_classifier(x_train_tf, train_df, x_test_tfidf):
    """
    classify data by cat boost classifier
    :param x_train_tf: training data represented as counted vector
    :param train_df: the training data
    :param x_test_tfidf: test data represented as counted vector
    :return: predicted labels
    """
    cat_features = [0, 1]
    model = CatBoostClassifier(iterations=2,
                               learning_rate=1,
                               depth=2)
    model.fit(x_train_tf, train_df.label, cat_features)
    predictions = model.predict(x_test_tfidf)

    return predictions


def get_all_classifiers_evaluations(data):
    train_df, test_df = prepare_data_for_classify(data)
    data_exploration(train_df)
    x_train_counts, x_test_counts = bag_of_words(train_df, test_df)
    x_train_tf, x_test_tfidf = tf_idf(x_train_counts, x_test_counts)

    prediction_M = majority_classifier(test_df)
    get_classifier_evaluation(prediction_M, test_df)

    prediction_NB = multinomial_naive_bayes_classifier(x_train_tf, train_df, x_test_tfidf)
    get_classifier_evaluation(prediction_NB, test_df)

    prediction_RL = regression_logistic_classifier(x_train_tf, train_df, x_test_tfidf)
    get_classifier_evaluation(prediction_RL, test_df)

    prediction_RF = random_forest_classifier(x_train_tf, train_df, x_test_tfidf)
    get_classifier_evaluation(prediction_RF, test_df)

    prediction_CB = cat_boost_classifier(x_train_tf, train_df, x_test_tfidf)
    get_classifier_evaluation(prediction_CB, test_df)


def get_classifier_evaluation(prediction, test, b=2):
    evaluation = Evaluation(prediction, test, b)
    evaluation.get_evaluation()





