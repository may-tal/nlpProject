import textStatistics as ts
from data import *
import classifiers
import textNormalization as tn
import clustering
import topicModeling as tm

PATH = 'Data/Data/clean'


def main():
    #d = read_data(['sentences.pos'], PATH)

    # text normalization (get the clean data)
    # d_norm = tn.text_normalization(d)
    # d_norm.to_csv('d_norm.csv', index=False, encoding='utf-8')

    d_norm = pd.read_csv('d_norm.csv')
    d_norm = tn.fix_yap(d_norm)
    d_norm = tn.get_text_non_stopwords(d_norm)
    d_norm_pos = d_norm[d_norm['label'] == 1]


    # # text statistics
    # ts.get_text_statistics(d_norm, "data")
    # ts.get_text_statistics(d_norm, "normData")
    #
    # # classified data
    # rf_score = classifiers.get_all_classifiers_evaluations(d_norm)
    # tn_rf_score = classifiers.get_all_classifiers_evaluations(d_norm)
    # classifiers.compare_our_result(rf_score, tn_rf_score)

    # K-means clustering
    clustering.get_class(d_norm_pos)

    # # topic modeling
    #     # vectors, vocab = tm.data_processing(d_norm)
    #     #
    #     # # show topics - SVD
    #     # u, s, vh = tm.svd(vectors)
    #     # print(tm.get_topics(vh[:6], vocab))
    #     # tm.print_messages_by_topic(d_norm, u)
    #     #
    #     # # show topic - NMF
    #     # w1, h1 = tm.nmf(vectors)
    #     # print(tm.get_topics(h1, vocab))

    # vectors, vocab = tm.data_processing(d_norm)
    # u, s, vh = tm.svd(vectors)
    # print(tm.show_topics(vh[:6], vocab))


if __name__ == "__main__":
    main()

