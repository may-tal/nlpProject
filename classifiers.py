import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from matplotlib.font_manager import FontProperties
from catboost import CatBoostClassifier
from evaluation import Evaluator
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import feature_extraction
import feature_selection
import lightgbm as lgb
import xgboost as xgb

BAG_OF_WORDS = 1
WORDS_2_VEC = 2
BOW_CHAR = 3

def majority_classifier(test_df):
    """
    classify data by majority classifier
    :param test_df: test data
    :return: predict labels
    """
    return len(test_df) * [0]


def get_majority_label_scores_and_prediction(test_df, label):
    """
    this function return the scores and prediction of majority classifier
    """
    prediction_M = majority_classifier(test_df)
    return get_classifier_evaluation(prediction_M, test_df, 'majority classifier', label), prediction_M


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


def get_multinomial_naive_bayes_scores_and_predication(x_train_tf, x_test_tfidf, train_df, test_df, label):
    """
    this function return the scores and prediction of naive bayes classifier
    """
    prediction_NB, predict_proba_NB = multinomial_naive_bayes_classifier(x_train_tf, train_df, x_test_tfidf)
    return get_classifier_evaluation(prediction_NB, test_df, 'naive bayes', label), prediction_NB, predict_proba_NB


def logistic_regression_classifier(x_train_tf, train_df, x_test_tfidf):
    """
    classify data by regression logistic classifier
    :param x_train_tf: training data represented as counted vector
    :param train_df: the training data
    :param x_test_tfidf: test data represented as counted vector
    :return: predicted labels
    """
    logistic_regression = LogisticRegression()
    clf = logistic_regression.fit(x_train_tf, train_df.label)
    predictions = logistic_regression.predict(x_test_tfidf)
    predict_proba = logistic_regression.predict_proba(x_test_tfidf)[:, 1]
    return predictions, predict_proba, clf, logistic_regression


def get_logistic_regression_scores_and_prediction(x_train_tf, x_test_tfidf, train_df, test_df, label):
    """
    this function return the scores and prediction of logistic regression classifier
    """
    prediction_LR, predict_proba_LR, clf, lr_model = logistic_regression_classifier(x_train_tf, train_df, x_test_tfidf)
    return get_classifier_evaluation(prediction_LR, test_df, 'logistic regression', label), prediction_LR, predict_proba_LR


def get_strongest_words(label, clf, traindf):
    """
    get the words that most influenced the classifier
    :param label: violence or not
    :param clf: classifier
    """
    cv = CountVectorizer()
    cv.fit_transform(traindf.text)
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
    return predictions, predict_proba, random_forest


def get_random_forest_scores_and_prediction(x_train_tf, x_test_tfidf, train_df, test_df, label, only_rf):
    """
    this function return the scores and prediction of random forest classifier
    """
    prediction_RF, predict_proba_RF, rf_model = random_forest_classifier(x_train_tf, train_df, x_test_tfidf)
    if only_rf:
        x_train_sel, selector = feature_selection.select_from_model(rf_model, x_train_tf, train_df)
        x_test_sel = selector.transform(x_test_tfidf)
        prediction_RF_new, predict_proba_RF_new, rf_new_model = random_forest_classifier(x_train_sel, train_df, x_test_sel)
        return get_classifier_evaluation(prediction_RF, test_df, 'random forest', label), prediction_RF, \
               predict_proba_RF, get_classifier_evaluation(prediction_RF_new, test_df, 'random forest', label)

    return get_classifier_evaluation(prediction_RF, test_df, 'random forest', label), prediction_RF, predict_proba_RF


def catboost_classifier(x_train_tf, train_df, x_test_tfidf):
    """
    classify data by cat boost classifier
    :param x_train_tf: training data represented as counted vector
    :param train_df: the training data
    :param x_test_tfidf: test data represented as counted vector
    :return: predicted labels
    """
    model = CatBoostClassifier(iterations=20)
    x_train_tf = pd.DataFrame(x_train_tf)
    x_test_tfidf = pd.DataFrame(x_test_tfidf)
    model.fit(x_train_tf, train_df.label)
    predictions = model.predict(x_test_tfidf)
    predict_proba = model.predict_proba(x_test_tfidf)
    return predictions, predict_proba


def lightgbm_classifier(x_train_tf, train_df, x_test_tfidf):
    """
    classify data by lightgbm classifier
    :param x_train_tf: training data represented as counted vector
    :param train_df: the training data
    :param x_test_tfidf: test data represented as counted vector
    :return: predicted labels
    """
    model = lgb.LGBMClassifier()
    model.fit(x_train_tf, train_df.label)
    predictions = model.predict(x_test_tfidf)
    predict_proba = model.predict_proba(x_test_tfidf)[:, 1]
    return predictions, predict_proba


def get_lightgbm_scores_and_prediction(x_train_tf, x_test_tfidf, train_df, test_df, label):
    """
    this function return the scores and prediction of lightgbmt classifier
    """
    prediction_LGB, prediction_proba_LGB = lightgbm_classifier(x_train_tf, train_df, x_test_tfidf)
    return get_classifier_evaluation(prediction_LGB, test_df, 'lightgbm', label), prediction_LGB, prediction_proba_LGB


def xgboost_classifier(x_train_tf, train_df, x_test_tfidf):
    """
    classify data by xgboost classifier
    :param x_train_tf: training data represented as counted vector
    :param train_df: the training data
    :param x_test_tfidf: test data represented as counted vector
    :return: predicted labels
    """
    model = xgb.XGBRFClassifier()
    model.fit(x_train_tf, train_df.label)
    predictions = model.predict(x_test_tfidf)
    predictions_proba = model.predict_proba(x_test_tfidf)[:, 1]
    return predictions, predictions_proba

def get_xgboost_scores_and_prediction(x_train_tf, x_test_tfidf, train_df, test_df, label):
    """
    this function return the scores and prediction of xgboost classifier
    """
    prediction_XGB, prediction_proba_XGB = xgboost_classifier(x_train_tf, train_df, x_test_tfidf)
    return get_classifier_evaluation(prediction_XGB, test_df, 'xgboost', label), prediction_XGB, prediction_proba_XGB


def get_all_classifiers_scores(x_train_tf, x_test_tfidf, train_df, test_df, label, only_rf):
    """
    this function return the scores of all the classifiers
    """
    scores = []
    scores_M, prediction_M = get_majority_label_scores_and_prediction(test_df, label)

    scores_NB, prediction_NB, predict_proba_NB = \
        get_multinomial_naive_bayes_scores_and_predication(x_train_tf, x_test_tfidf, train_df, test_df, label)

    scores_LR, prediction_LR, predict_proba_LR = \
        get_logistic_regression_scores_and_prediction(x_train_tf, x_test_tfidf, train_df, test_df, label)

    scores_LGB, prediction_LGB, prediction_proba_LGB = \
        get_lightgbm_scores_and_prediction(x_train_tf, x_test_tfidf, train_df, test_df, label)

    scores_XGB, prediction_XGB, prediction_proba_XGB = \
        get_xgboost_scores_and_prediction(x_train_tf, x_test_tfidf, train_df, test_df, label)

    if only_rf:
        scores_RF, prediction_RF, predict_proba_RF, score_RF_new = \
            get_random_forest_scores_and_prediction(x_train_tf, x_test_tfidf, train_df, test_df, label, only_rf)
        scores.append(scores_RF)
        scores.append(score_RF_new)
    else:
        scores_RF, prediction_RF, predict_proba_RF = \
            get_random_forest_scores_and_prediction(x_train_tf, x_test_tfidf, train_df, test_df, label, only_rf)
        plot_roc_curve(test_df, prediction_M, predict_proba_NB, predict_proba_LR, predict_proba_RF, prediction_proba_LGB, prediction_proba_XGB)
        scores.append(scores_M)
        scores.append(scores_NB)
        scores.append(scores_LR)
        scores.append(scores_RF)
        scores.append(scores_LGB)
        scores.append(scores_XGB)

    plot_table_scores(scores, label, only_rf)

    return scores


def plot_scores_by_feature_extraction(data, title, flag, only_rf):
    """
    this function return the table score of all the classifiers by feature extraction
    """
    if flag == BAG_OF_WORDS:
        x_train_tf, x_test_tfidf, train_df, test_df = feature_extraction.get_bow_tfidf(data, True)
        scores = get_all_classifiers_scores(x_train_tf, x_test_tfidf, train_df, test_df, "\n(bag of words) "
                                            + title, only_rf)

    elif flag == BOW_CHAR:
        x_train_tf, x_test_tfidf, train_df, test_df = feature_extraction.get_bow_tfidf(data, False)
        scores = get_all_classifiers_scores(x_train_tf, x_test_tfidf, train_df, test_df,
                                            "\n(bow character level n grams) " + title, only_rf)

    else:  # WORDS_2_VEC
        x_train_tf, x_test_tfidf, train_df, test_df = feature_extraction.get_word2vec(data)
        scores = get_all_classifiers_scores(x_train_tf, x_test_tfidf, train_df, test_df, "\n(word2vec) " + title, only_rf)

    return scores


def plot_table_scores(scores, title, only_rf):
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
    if only_rf:
        rows_labels = ['random forest', 'random forest\nwith feature selection']
    else:
        rows_labels = ['majority', 'naive bayes', 'regression logistic', 'random forest', 'lightgbm', 'xgboost ']
    ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellColours=colours, rowLabels=rows_labels)
    fig.tight_layout()
    plt.savefig("scores.png")
    plt.title("classifier_evaluation " + title)
    plt.show()


def get_classifier_evaluation(prediction, test, classifier_name, data_name, b=2):
    """
    this function get the evaluation of each classifier: print the amount of errors and the text of them, plot roc_curve
    and return the measures scores.
    """
    evaluation = Evaluator(prediction, test, b)
    return evaluation.get_evaluation(classifier_name, data_name)


def helper_plot_curve(test, pred, pos_label, classifier_name, curve_name):
    """
    helper function for plot_curve
    """
    fpr, tpr, _ = roc_curve(test['label'], pred, pos_label=pos_label)
    plt.plot(fpr, tpr, lw=2, label=classifier_name + '- ' + curve_name + ' curve')


def plot_curve(test, pred_m, pred_nb, pred_rl, pred_rf, pred_lgb, pred_xgb, pos_label, curve_name):
    """
    this function plot the curve
    """
    helper_plot_curve(test, pred_m, pos_label, 'Majority', curve_name)
    helper_plot_curve(test, pred_nb, pos_label, 'Naive bayes', curve_name)
    helper_plot_curve(test, pred_rl, pos_label, 'Regression logistic', curve_name)
    helper_plot_curve(test, pred_rf, pos_label, 'Random forest', curve_name)
    helper_plot_curve(test, pred_lgb, pos_label, 'Lightgbm', curve_name)
    helper_plot_curve(test, pred_xgb, pos_label, 'Xgboost', curve_name)
    plt.title(curve_name + " CURVE")
    plt.ylabel("true positive rate")
    plt.xlabel("false positive rate")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.legend(loc="lower right")
    plt.show()


def plot_roc_curve(test, pred_m, pred_nb, pred_rl, pred_rf, pred_lgb, pred_xgb):
    """
    this function plot the roc curve
    """
    plot_curve(test, pred_m, pred_nb, pred_rl, pred_rf, pred_lgb, pred_xgb, 1, 'ROC')


def plot_nr_curve(test, pred_m, pred_nb, pred_rl, pred_rf, pred_lgb, pred_xgb):
    """
    this function plot the nr curve
    """
    plot_curve(test, pred_m, pred_nb, pred_rl, pred_rf, pred_lgb, pred_xgb, 0, 'NL')


def compare_our_result(rf, tn_rf):
    """
    compare results of method stages
    :param rf: random forest stage results
    :param tn_rf: text normalization and random forest results
    :return: None, build graphs
    """
    # convert to percentages
    rf = [x * 100 for x in rf]
    tn_rf = [x * 100 for x in tn_rf]

    plt.clf()

    # set width of bar
    bar_width = 0.10

    # Set position of bar on X axis
    r1 = np.arange(len(rf))
    r2 = [x + bar_width for x in r1]

    # Make the plot
    plt.bar(r1, rf, color='orange', width=bar_width, edgecolor='white', label='RF')
    plt.bar(r2, tn_rf, color='blue', width=bar_width, edgecolor='white', label='text normalization + RF')


    plt.ylabel('Precision percentage', fontweight='bold')
    # Add xticks on the middle of the group bars
    plt.xticks([r + bar_width for r in range(len(rf))], ['precision', 'recall', 'accuracy', 'F1', 'F2'])

    plt.title("Compare stages result")

    # limit the graph
    plt.ylim(bottom=0, top=104.9)

    # Create legend & Save graphic
    font_p = FontProperties()
    font_p.set_size('small')
    plt.legend(loc='upper left', prop=font_p)

    plt.savefig("Compare stages result.png")
    plt.show()





