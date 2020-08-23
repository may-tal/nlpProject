from sklearn.feature_selection import SelectFromModel
import feature_extraction
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer


def removing_features_with_low_variance(x_train_tf):
    """
    VarianceThreshold is a simple baseline approach to feature selection.
    It removes all features whose variance doesn’t meet some threshold.
    By default, it removes all zero-variance features
    :param x_train_tf: the train data without labels
    :return: x_train_sel (the data after selection) and the selector
    """
    selector = VarianceThreshold()
    x_train_sel = selector.fit_transform(x_train_tf)
    return x_train_sel, selector


def univariate_feature_selection(x_train_tf, train_df):
    """
    Univariate feature selection works by selecting the best features
    based on univariate statistical tests. It can be seen as a
    preprocessing step to an estimator.
    We perform a chi2 test to the samples to retrieve only the 800 best features
    :param x_train_tf: the train data without labels
    :param train_df: the train data
    :return: x_train_sel (the data after selection) and the selector
    """
    selector = SelectKBest(chi2, k=800)
    x_train_sel = selector.fit_transform(x_train_tf, train_df.label)
    return x_train_sel, selector


def select_from_model(estimator, x_train_tf, train_df):
    """
    SelectFromModel is a meta-transformer that can be used along with any estimator that has a
    coef_ or feature_importances_ attribute after fitting. The features are considered
    unimportant and removed, if the corresponding coef_ or feature_importances_
    values are below the provided threshold parameter
    :param estimator: the classification model
    :param x_train_tf:  the train data without labels
    :param train_df: the train data
    :return: x_train_sel (the data after selection) and the selector
    """
    model = estimator.fit(x_train_tf, train_df.label)
    selector = SelectFromModel(estimator=model, prefit=True)
    x_train_sel = selector.transform(x_train_tf)

    return x_train_sel, selector












