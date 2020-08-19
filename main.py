import textStatistics as ts
from data import *
import classifiers
import textNormalization as tn
import clustering
import topicModeling as tm

PATH = 'Data/Data/clean'


def main():
    orig_data = pd.read_csv('orig_data.csv')
    yap_data = tn.remove_punctuation(orig_data)
    yap_data = tn.stemmer_and_lemmatizer(yap_data)
    yap_data.to_csv('yap_data.csv', index=False, encoding='utf-8')

    # orig_data, clean_data, clean_and_normalization_data_without_yap, normalization_data = get_data_as_csv()
    #
    # classifiers.plot_scores_by_feature_extraction(orig_data, "orig data")
    # classifiers.plot_scores_by_feature_extraction(clean_data, "clean data")
    # classifiers.plot_scores_by_feature_extraction(clean_and_normalization_data_without_yap, "clean + norm - yap")
    # classifiers.plot_scores_by_feature_extraction(normalization_data, "norm data")


    # normalization_data_pos = normalization_data[normalization_data['label'] == 1]
    #
    # # text statistics
    # ts.get_text_statistics(normalization_data, "data")
    # ts.get_text_statistics(normalization_data, "normData")
    #
    # classified data
    # scores = classifiers.get_all_classifiers_evaluations(orig_data)
    # tn_scores = classifiers.get_all_classifiers_evaluations(normalization_data)
    # classifiers.compare_our_result(scores, tn_scores)
    #
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

    # data_without_punctuation_with_yap = tn.get_data_without_punctuation_with_yap(orig_data)
    # data_without_punctuation_with_yap.to_csv('data_without_punctuation_with_yap.csv', index=False, encoding='utf-8')

    clean_and_normalization_data_without_yap = tn.text_normaliztion_without_yap(orig_data)
    clean_and_normalization_data_without_yap.to_csv('clean_and_normalization_data_without_yap.csv', index=False,
                                                    encoding='utf-8')

    # normalization_data = tn.text_normalization(orig_data)
    # normalization_data.to_csv('normalization_data.csv', index=False, encoding='utf-8')

    return orig_data, clean_data, clean_and_normalization_data_without_yap

def get_data_as_csv():
    orig_data = pd.read_csv('orig_data.csv')
    clean_data = pd.read_csv('clean_data.csv')
    clean_and_normalization_data_without_yap = pd.read_csv('clean_and_normalization_data_without_yap.csv')
    normalization_data = pd.read_csv('d_norm.csv')
    return orig_data, clean_data, clean_and_normalization_data_without_yap, normalization_data

if __name__ == "__main__":
    main()

