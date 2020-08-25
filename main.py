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
    orig_data, clean_data, clean_and_normalization_data_without_yap, yap_no_punct, normalization_data = get_data_as_csv()
    #clean_data = data_to_csv()

    # plot the scores of the best data
    clean_data_scores = classifiers.plot_scores_by_feature_extraction(clean_data, "clean data", BOW_CHAR, False)
    norm_data_scores = classifiers.plot_scores_by_feature_extraction(normalization_data, "normalization data", BAG_OF_WORDS, False)

    # compare our result
    scores = classifiers.plot_scores_by_feature_extraction(orig_data, "original data", BOW_CHAR, False)
    classifiers.compare_our_result(clean_data_scores, norm_data_scores)

    # # plot the scores of the best data with feature selection - random forest classifier
    # clean_data_scores = classifiers.plot_scores_by_feature_extraction(clean_data, "clean data", BOW_CHAR, True)
    # norm_data_scores = classifiers.plot_scores_by_feature_extraction(normalization_data, "normalization data", BAG_OF_WORDS, True)
    #
    # # random forest - word2vec
    # clean_data_scores = classifiers.plot_scores_by_feature_extraction(yap_no_punct, "clean data", WORD2VEC, True)
    # norm_data_scores = classifiers.plot_scores_by_feature_extraction(normalization_data, "normalization data", WORD2VEC, True)

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

    data_without_punctuation_with_yap = tn.get_data_without_punctuation_with_yap(orig_data)
    data_without_punctuation_with_yap.to_csv('data_without_punctuation_with_yap.csv', index=False, encoding='utf-8')

    clean_and_normalization_data_without_yap = tn.text_normaliztion_without_yap(orig_data)
    clean_and_normalization_data_without_yap.to_csv('clean_and_normalization_data_without_yap.csv', index=False,
                                                    encoding='utf-8')

    normalization_data = tn.text_normalization(orig_data)
    normalization_data.to_csv('normalization_data.csv', index=False, encoding='utf-8')

    return orig_data, clean_data, data_without_punctuation_with_yap, clean_and_normalization_data_without_yap,\
           data_without_punctuation_with_yap


def get_data_as_csv():
    orig_data = pd.read_csv('orig_data.csv')
    clean_data = pd.read_csv('clean_data.csv')
    clean_and_normalization_data_without_yap = pd.read_csv('clean_and_normalization_data_without_yap.csv')
    yap_no_punct = pd.read_csv('data_without_punctuation_with_yap.csv')
    normalization_data = pd.read_csv('d_norm.csv')
    return orig_data, clean_data, clean_and_normalization_data_without_yap, yap_no_punct, normalization_data


if __name__ == "__main__":
    main()

