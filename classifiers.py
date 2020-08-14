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
    # x_train_sel, selector = feature_selection.select_from_model(logistic_regression, x_train_tf, train_df)
    # x_test_sel = selector.transform(x_test_tfidf)
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
    x_train_tf, x_test_tfidf, train_df, test_df = feature_extraction.get_bow_tfidf(data)

    scores = []

    # prediction_M = majority_classifier(test_df)
    # print("==========majority_classifier=============")
    # scores.append(get_classifier_evaluation(prediction_M, test_df, 'majority classifier'))
    #
    # prediction_NB, predict_proba_NB = multinomial_naive_bayes_classifier(x_train_tf, train_df, x_test_tfidf)
    # print("\n==========multinomial_naive_bayes_classifier=============")
    # scores.append(get_classifier_evaluation(prediction_NB, test_df, 'naive bayes'))
    #
    prediction_LR, predict_proba_LR, clf, lr_model = logistic_regression_classifier(x_train_tf, train_df, x_test_tfidf)
    print("\n==========logistic_regression_classifier=============")
    scores.append(get_classifier_evaluation(prediction_LR, test_df, 'logistic regression'))
    print("before selection: " + str(lr_model.score(x_test_tfidf, test_df.label)))
    get_strongest_words(0, clf, train_df)

    x_train_sel, selector = feature_selection.select_from_model(lr_model, x_train_tf, train_df)
    x_test_sel = selector.transform(x_test_tfidf)

    lr_new_model = LogisticRegression().fit(x_train_sel, train_df.label)
    print("after selection: " + str(lr_new_model.score(x_test_sel, test_df.label)))
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


    # prediction_RF, predict_proba_RF, rf_model = random_forest_classifier(x_train_tf, train_df, x_test_tfidf)
    # print("\n==========random_forest_classifier=============")
    # rf_scores = get_classifier_evaluation(prediction_RF, test_df, 'random forest')
    # scores.append(rf_scores)
    # print("before selection: " + str(rf_model.score(x_test_tfidf, test_df.label)))
    #
    # x_train_sel, selector = feature_selection.select_from_model(rf_model, x_train_tf, train_df)
    # x_test_sel = selector.transform(x_test_tfidf)
    #
    # rf_new_model = RandomForestClassifier().fit(x_train_sel, train_df.label)
    # print("after selection: " + str(rf_new_model.score(x_test_sel, test_df.label)))

    # prediction_CB, predict_proba_CB = cat_boost_classifier(x_train_tf, train_df, x_test_tfidf)
    # scores.append(get_classifier_evaluation(prediction_CB, test_df))

    # plot_roc_curve(test_df, prediction_M, predict_proba_NB, predict_proba_LR, predict_proba_RF)
    # plot_nr_curve(test_df, prediction_M, predict_proba_NB, predict_proba_LR, predict_proba_RF)
    plot_table_scores(scores)

    # return rf_scores


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
    #rows_labels = ['majority', 'naive bayes', 'regression logistic', 'random forest']
    rows_labels = ['rf']
    ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellColours=colours, rowLabels=rows_labels)
    fig.tight_layout()
    plt.savefig("scores.png")
    plt.show()


def get_classifier_evaluation(prediction, test, classifier_name, b=2):
    """
    this function get the evaluation of each classifier: print the amount of errors and the text of them, plot roc_curve
    and return the measures scores.
    """
    evaluation = Evaluator(prediction, test, b)
    return evaluation.get_evaluation(classifier_name)


def plot_roc_curve(test, pred_m, pred_nb, pred_rl, pred_rf):
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

    # fpr_cb, tpr_cb, _ = roc_curve(test['label'], pred_cb, pos_label=1)
    # roc_auc_cb = roc_auc_score(test['label'], pred_cb)
    # plt.plot(fpr_cb, tpr_cb, lw=2, label='Cat boost- ROC curve (area = %0.2f)' % roc_auc_cb)

    plt.title("ROC CURVE")
    plt.ylabel("true positive rate")
    plt.xlabel("false positive rate")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.legend(loc="lower right")
    plt.show()


def plot_nr_curve(test, pred_m, pred_nb, pred_rl, pred_rf):
    """
    this function plot the roc curve
    """
    fpr_m, tpr_m, _ = roc_curve(test['label'], pred_m, pos_label=0)
    roc_auc_m = roc_auc_score(test['label'], pred_m)
    plt.plot(fpr_m, tpr_m, lw=2, label='Majority- ROC curve (area = %0.2f)' % roc_auc_m)

    fpr_nb, tpr_nb, _ = roc_curve(test['label'], pred_nb, pos_label=0)
    roc_auc_nb = roc_auc_score(test['label'], pred_nb)
    plt.plot(fpr_nb, tpr_nb, lw=2, label='Naive bayes- ROC curve (area = %0.2f)' % roc_auc_nb)

    fpr_rl, tpr_rl, _ = roc_curve(test['label'], pred_rl, pos_label=0)
    roc_auc_rl = roc_auc_score(test['label'], pred_rl)
    plt.plot(fpr_rl, tpr_rl, lw=2, label='Logistic regression - ROC curve (area = %0.2f)' % roc_auc_rl)

    fpr_rf, tpr_rf, _ = roc_curve(test['label'], pred_rf, pos_label=0)
    roc_auc_rf = roc_auc_score(test['label'], pred_rf)
    plt.plot(fpr_rf, tpr_rf, lw=2, label='Random forest- ROC curve (area = %0.2f)' % roc_auc_rf)

    # fpr_cb, tpr_cb, _ = roc_curve(test['label'], pred_cb, pos_label=1)
    # roc_auc_cb = roc_auc_score(test['label'], pred_cb)
    # plt.plot(fpr_cb, tpr_cb, lw=2, label='Cat boost- ROC curve (area = %0.2f)' % roc_auc_cb)

    plt.title(" CURVE")
    plt.ylabel("false negative rate")
    plt.xlabel("true negative rate")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.legend(loc="lower right")
    plt.show()


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





