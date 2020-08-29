import text_statistics as ts
from data import *
import classifiers
import text_normalization as tn
import clustering
import topic_modeling as tm
import data

PATH = 'Data'
BAG_OF_WORDS = 1
WORD2VEC = 2
BOW_CHAR = 3

def main():
    # read the csv data files
    orig_data, clean_data, yap_punc_data, norm_no_yap_data, norm_data = get_data_as_csv()

    # plot the scores of the best data (clean data and normalization data) for each classifier
    clean_data_scores = classifiers.plot_scores_by_feature_extraction(clean_data, "clean data", BOW_CHAR, False, False)
    norm_data_scores = classifiers.plot_scores_by_feature_extraction(norm_data, "normalization data", BAG_OF_WORDS,
                                                                     False, False)

    # plot the scores of the best data with feature selection - random forest classifier
    clean_rf_scores_feature_selection = classifiers.plot_scores_by_feature_extraction(clean_data, "clean data", BOW_CHAR, True, True)
    norm_rf_scores_feature_selection = classifiers.plot_scores_by_feature_extraction(norm_data, "normalization data", BAG_OF_WORDS, True, True)

    # plot random forest scores with word2vec
    clean_data_scores = classifiers.plot_scores_by_feature_extraction(yap_punc_data, "clean data", WORD2VEC, True, False)
    norm_data_scores = classifiers.plot_scores_by_feature_extraction(norm_data, "normalization data", WORD2VEC, True, False)

    # normalization_data_posnormalization_data_pos = normalization_data[normalization_data['label'] == 1]
    # # K-means clustering
    # clustering.get_class(normalization_data_pos)
    #
    # # topic modeling
    # vectors, vocab = tm.data_processing(normalization_data)
    #
    # # show topics - SVD
    # u, s, vh = tm.svd(vectors)
    # print(tm.get_topics(vh[:6], vocab))
    # tm.print_messages_by_topic(normalization_data, u)
    #
    # # show topic - NMF
    # w1, h1 = tm.nmf(vectors)
    # print(tm.get_topics(h1, vocab))
    #
    # vectors, vocab = tm.data_processing(normalization_data)
    # u, s, vh = tm.svd(vectors)
    # print(tm.show_topics(vh[:6], vocab))


def data_to_csv():
    """
    this function make csv file of the original data, clean data, normalization data and ect..
    """
    orig_data = read_data(['sentences.pos', 'sentences.neg'], PATH)
    orig_data.to_csv('orig_data.csv', index=False, encoding='utf-8')

    clean_data = tn.get_clean_data(orig_data)
    clean_data.to_csv('clean_data.csv', index=False, encoding='utf-8')

    yap_punc_data = tn.get_data_without_punctuation_with_yap(orig_data)
    yap_punc_data.to_csv('yap_punc_data.csv', index=False, encoding='utf-8')

    norm_no_yap_data = tn.text_normaliztion_without_yap(orig_data)
    norm_no_yap_data.to_csv('norm_no_yap_data.csv', index=False,
                                                    encoding='utf-8')

    norm_data = tn.text_normalization(orig_data)
    norm_data.to_csv('norm_data.csv', index=False, encoding='utf-8')

    return orig_data, clean_data, yap_punc_data, norm_no_yap_data, norm_data


def get_data_as_csv():
    """
    this function read the csv files and return them.
    """
    orig_data = pd.read_csv('orig_data.csv')
    clean_data = pd.read_csv('clean_data.csv')
    norm_no_yap_data = pd.read_csv('norm_no_yap_data.csv')
    yap_punc_data = pd.read_csv('yap_punc_data.csv')
    norm_data = pd.read_csv('norm_data.csv')
    return orig_data, clean_data, norm_no_yap_data, yap_punc_data, norm_data


if __name__ == "__main__":
    main()

