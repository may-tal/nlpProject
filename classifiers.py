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


def get_all_classifiers_scores(x_train_tf, x_test_tfidf, train_df, test_df, label):
    scores = []

    prediction_M = majority_classifier(test_df)
    scores.append(get_classifier_evaluation(prediction_M, test_df, 'majority classifier'))

    prediction_NB, predict_proba_NB = multinomial_naive_bayes_classifier(x_train_tf, train_df, x_test_tfidf)
    scores.append(get_classifier_evaluation(prediction_NB, test_df, 'naive bayes'))

    prediction_LR, predict_proba_LR, clf, lr_model = logistic_regression_classifier(x_train_tf, train_df, x_test_tfidf)
    scores.append(get_classifier_evaluation(prediction_LR, test_df, 'logistic regression'))

    prediction_RF, predict_proba_RF, rf_model = random_forest_classifier(x_train_tf, train_df, x_test_tfidf)
    scores.append(get_classifier_evaluation(prediction_RF, test_df, 'random forest'))

    prediction_LGB, prediction_proba_LGB = lightgbm_classifier(x_train_tf, train_df, x_test_tfidf)
    scores.append(get_classifier_evaluation(prediction_LGB, test_df, 'lightgbm'))

    prediction_XGB, prediction_proba_XGB = xgboost_classifier(x_train_tf, train_df, x_test_tfidf)
    scores.append(get_classifier_evaluation(prediction_XGB, test_df, 'xgboost'))

    # plot_roc_curve(test_df, prediction_M, predict_proba_NB, predict_proba_LR, predict_proba_RF, prediction_proba_LGB, prediction_proba_XGB)
    # plot_nr_curve(test_df, prediction_M, predict_proba_NB, predict_proba_LR, predict_proba_RF, prediction_proba_LGB, prediction_proba_XGB)
    plot_table_scores(scores, label)

    return scores


def plot_scores_by_feature_extraction(data, title):
    # bag of words
    x_train_tf, x_test_tfidf, train_df, test_df = feature_extraction.get_bow_tfidf(data, True)
    scores = get_all_classifiers_scores(x_train_tf, x_test_tfidf, train_df, test_df, "(bag of words) " + title)

    # bow character level n grams
    x_train_tf, x_test_tfidf, train_df, test_df = feature_extraction.get_bow_tfidf(data, False)
    scores = get_all_classifiers_scores(x_train_tf, x_test_tfidf, train_df, test_df, "(bow character level n grams) " + title)

    # word2vec
    x_train_tf, x_test_tfidf, train_df, test_df = feature_extraction.get_word2vec(data)
    scores = get_all_classifiers_scores(x_train_tf, x_test_tfidf, train_df, test_df, "(word2vec) " + title)


def get_all_classifiers_evaluations(data):
    """
    this function print and plot for each classifier his evaluation
    """
    # bag of words
    x_train_tf, x_test_tfidf, train_df, test_df = feature_extraction.get_bow_tfidf(data, True)
    # plot classifier scores, roc curve, nr curve and get all classifiers scores
    scores = get_all_classifiers_scores(x_train_tf, x_test_tfidf, train_df, test_df, "bag of words")

    # x_train_sel, selector = feature_selection.select_from_model(lr_model, x_train_tf, train_df)
    # x_test_sel = selector.transform(x_test_tfidf)
    #
    # lr_new_model = LogisticRegression().fit(x_train_sel, train_df.label)
    # print("after selection: " + str(lr_new_model.score(x_test_sel, test_df.label)))
    #
    # x_train_sel_2, selector2= feature_selection.removing_features_with_low_variance(x_train_tf)
    # x_test_sel2 = selector2.transform(x_test_tfidf)
    # lr_new_model_2 = LogisticRegression().fit(x_train_sel_2, train_df.label)
    # print("after selection: " + str(lr_new_model_2.score(x_test_sel2, test_df.label)))
    #
    # x_train_sel_3, selector3 = feature_selection.univariate_feature_selection(x_train_tf, train_df)
    # x_test_sel3 = selector3.transform(x_test_tfidf)
    # lr_new_model_3 = LogisticRegression().fit(x_train_sel_3, train_df.label)
    # print("after selection: " + str(lr_new_model_3.score(x_test_sel3, test_df.label)))

    # rf_scores = get_classifier_evaluation(prediction_RF, test_df, 'random forest')
    # scores.append(rf_scores)
    # print("before selection: " + str(rf_model.score(x_test_tfidf, test_df.label)))
    #
    # x_train_sel, selector = feature_selection.select_from_model(rf_model, x_train_tf, train_df)
    # x_test_sel = selector.transform(x_test_tfidf)
    #
    # rf_new_model = RandomForestClassifier().fit(x_train_sel, train_df.label)
    # print("after selection: " + str(rf_new_model.score(x_test_sel, test_df.label)))

    return scores


def plot_table_scores(scores, title):
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
    rows_labels = ['majority', 'naive bayes', 'regression logistic', 'random forest', 'lightgbm', 'xgboost ']
    ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellColours=colours, rowLabels=rows_labels)
    fig.tight_layout()
    plt.savefig("scores.png")
    plt.title("classifier_evaluation " + title)
    plt.show()


def get_classifier_evaluation(prediction, test, classifier_name, b=2):
    """
    this function get the evaluation of each classifier: print the amount of errors and the text of them, plot roc_curve
    and return the measures scores.
    """
    evaluation = Evaluator(prediction, test, b)
    return evaluation.get_evaluation(classifier_name)


def helper_plot_curve(test, pred, pos_label, classifier_name):
    """
    helper function for plot_curve
    """
    fpr, tpr, _ = roc_curve(test['label'], pred, pos_label=pos_label)
    roc_auc = roc_auc_score(test['label'], pred)
    plt.plot(fpr, tpr, lw=2, label=classifier_name+'- ROC curve (area = %0.2f)' % roc_auc)


def plot_curve(test, pred_m, pred_nb, pred_rl, pred_rf, pred_lgb, pred_xgb, pos_label):
    """
    this function plot the curve
    """
    helper_plot_curve(test, pred_m, pos_label, 'Majority')
    helper_plot_curve(test, pred_nb, pos_label, 'Naive bayes')
    helper_plot_curve(test, pred_rl, pos_label, 'Regression logistic')
    helper_plot_curve(test, pred_rf, pos_label, 'Random forest')
    helper_plot_curve(test, pred_lgb, pos_label, 'Lightgbm')
    helper_plot_curve(test, pred_xgb, pos_label, 'Xgboost')
    plt.title("ROC CURVE")
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
    plot_curve(test, pred_m, pred_nb, pred_rl, pred_rf, pred_lgb, pred_xgb, 1)


def plot_nr_curve(test, pred_m, pred_nb, pred_rl, pred_rf, pred_lgb, pred_xgb):
    """
    this function plot the nr curve
    """
    plot_curve(test, pred_m, pred_nb, pred_rl, pred_rf, pred_lgb, pred_xgb, 0)


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





