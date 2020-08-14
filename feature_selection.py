from sklearn.feature_selection import SelectFromModel
import feature_extraction
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer


def removing_features_with_low_variance(x_train_tf):
    selector = VarianceThreshold()
    x_train_sel = selector.fit_transform(x_train_tf)
    return x_train_sel, selector


def univariate_feature_selection(x_train_tf, train_df):
    selector = SelectKBest(chi2, k=800)
    x_train_sel = selector.fit_transform(x_train_tf, train_df.label)
    return x_train_sel, selector


def select_from_model(estimator, x_train_tf, train_df):
    # vectorizer = CountVectorizer(lowercase=False)
    # vectorizer.fit_transform(train_df.text)

    model = estimator.fit(x_train_tf, train_df.label)
    selector = SelectFromModel(estimator=model, prefit=True)
    x_train_sel = selector.transform(x_train_tf)
    # mask = selector.get_support()
    # words = vectorizer.get_feature_names()
    #
    # selected_words = [word for word, is_taken in zip(words, mask) if is_taken]
    # print(selected_words)

    return x_train_sel, selector












