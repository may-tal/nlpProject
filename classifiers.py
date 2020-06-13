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
from evaluation import Evaluator
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


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
    naive_bayes = MultinomialNB()
    naive_bayes.fit(x_train_tf, train_df.label)
    predictions = naive_bayes.predict(x_test_tfidf)
    predict_proba = naive_bayes.predict_proba(x_test_tfidf)[:, 1]
    return predictions, predict_proba

def regression_logistic_classifier(x_train_tf, train_df, x_test_tfidf):
    """
    classify data by regression logistic classifier
    :param x_train_tf: training data represented as counted vector
    :param train_df: the training data
    :param x_test_tfidf: test data represented as counted vector
    :return: predicted labels
    """
    regression_logistic = LogisticRegression()
    regression_logistic.fit(x_train_tf, train_df.label)
    predictions = regression_logistic.predict(x_test_tfidf)
    predict_proba = regression_logistic.predict_proba(x_test_tfidf)[:, 1]
    return predictions, predict_proba


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
    random_forest = RandomForestClassifier()
    random_forest.fit(x_train_tf, train_df.label)
    predictions = random_forest.predict(x_test_tfidf)
    predict_proba = random_forest.predict_proba(x_test_tfidf)[:, 1]
    return predictions, predict_proba


def cat_boost_classifier(x_train_tf, train_df, x_test_tfidf):
    """
    classify data by cat boost classifier
    :param x_train_tf: training data represented as counted vector
    :param train_df: the training data
    :param x_test_tfidf: test data represented as counted vector
    :return: predicted labels
    """
    cat_features = [0, 1]
    model = CatBoostClassifier(iterations=1)
    # model.fit(x_train_tf, train_df.label, cat_features)
    model.fit(x_train_tf, train_df.label)
    predictions = model.predict(x_test_tfidf)
    #TODO- check why predict proba is 2dim array 
    predict_proba = model.predict_proba(x_test_tfidf)
    return predictions, predict_proba


def get_all_classifiers_evaluations(data):
    """
    this function print and plot for each classifier his evaluation
    """
    train_df, test_df = prepare_data_for_classify(data)
    data_exploration(train_df)
    x_train_counts, x_test_counts = bag_of_words(train_df, test_df)
    x_train_tf, x_test_tfidf = tf_idf(x_train_counts, x_test_counts)

    scores = []

    prediction_M = majority_classifier(test_df)
    scores.append(get_classifier_evaluation(prediction_M, test_df))

    prediction_NB, predict_proba_NB = multinomial_naive_bayes_classifier(x_train_tf, train_df, x_test_tfidf)
    scores.append(get_classifier_evaluation(prediction_NB, test_df))

    prediction_RL, predict_proba_RL = regression_logistic_classifier(x_train_tf, train_df, x_test_tfidf)
    scores.append(get_classifier_evaluation(prediction_RL, test_df))

    prediction_RF, predict_proba_RF = random_forest_classifier(x_train_tf, train_df, x_test_tfidf)
    scores.append(get_classifier_evaluation(prediction_RF, test_df))

    prediction_CB, predict_proba_CB = cat_boost_classifier(x_train_tf, train_df, x_test_tfidf)
    scores.append(get_classifier_evaluation(prediction_CB, test_df))

    plot_roc_curve(test_df, prediction_M, predict_proba_NB, predict_proba_RL, predict_proba_RF, predict_proba_CB)
    plot_table_scores(scores)


def plot_table_scores(scores):
    """
    this function plot the scores of each classifier (recall, precision,...)
    """
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    df = pd.DataFrame(scores, columns=['Recall score', 'Precision score', 'Accuracy score', 'F1 score', 'F2 score'])
    vals = np.around(df.values, 2)
    norm = plt.Normalize(vals.min() - 1, vals.max() + 1)
    colours = plt.cm.hot(norm(vals))
    rows_labels = ['majority', 'naive bayes', 'regression logistic', 'random forest', 'cat boost']
    ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellColours=colours, rowLabels=rows_labels)
    fig.tight_layout()
    plt.savefig("scores.png")
    plt.show()


def get_classifier_evaluation(prediction, test, b=2):
    """
    this function get the evaluation of each classifier: print the amount of errors and the text of them, plot roc_curve
    and return the measures scores.
    """
    evaluation = Evaluator(prediction, test, b)
    return evaluation.get_evaluation()


def plot_roc_curve(test, pred_m, pred_nb, pred_rl, pred_rf, pred_cb):
    """
    this function plot the roc curve
    """
    fpr_m, tpr_m, _ = roc_curve(test['label'], pred_m, pos_label=1)
    roc_auc_m = roc_auc_score(test['label'], pred_m)
    plt.plot(fpr_m, tpr_m, lw=2, label='Majority- ROC curve (area = %0.2f)' % roc_auc_m)

    fpr_nb, tpr_nb, _ = roc_curve(test['label'], pred_nb, pos_label=1)
    roc_auc_nb = roc_auc_score(test['label'], pred_nb)
    plt.plot(fpr_nb, tpr_nb, lw=2, label='Naive bayes- ROC curve (area = %0.2f)' % roc_auc_nb)

    fpr_rl, tpr_rl, _ = roc_curve(test['label'], pred_rl, pos_label=1)
    roc_auc_rl = roc_auc_score(test['label'], pred_rl)
    plt.plot(fpr_rl, tpr_rl, lw=2, label='Regression logistic- ROC curve (area = %0.2f)' % roc_auc_rl)

    fpr_rf, tpr_rf, _ = roc_curve(test['label'], pred_rf, pos_label=1)
    roc_auc_rf = roc_auc_score(test['label'], pred_rf)
    plt.plot(fpr_rf, tpr_rf, lw=2, label='Random forest- ROC curve (area = %0.2f)' % roc_auc_rf)

    fpr_cb, tpr_cb, _ = roc_curve(test['label'], pred_cb, pos_label=1)
    roc_auc_cb = roc_auc_score(test['label'], pred_cb)
    plt.plot(fpr_cb, tpr_cb, lw=2, label='Cat boost- ROC curve (area = %0.2f)' % roc_auc_cb)

    plt.title("ROC CURVE")
    plt.ylabel("true positive rate")
    plt.xlabel("false positive rate")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.legend(loc="lower right")
    plt.show()





