import textStatistics as ts
from data import *
import classifiers
import textNormalization as tn
<<<<<<< Updated upstream
import clustering
=======
import topicModeling as tm
>>>>>>> Stashed changes

PATH = 'Data/Data/clean'


def main():
    # d = read_data(['sentences.neg', 'sentences.pos'], PATH)

    # text normalization (get the clean data)
    # d_norm = tn.text_normalization(d)
    # d_norm.to_csv('d_norm.csv', index=False, encoding='utf-8')

    d_norm = pd.read_csv('d_norm.csv')
    d_norm = tn.get_text_non_stopwords(d_norm)

    # text statistics
    # ts.get_text_statistics(d, "data")
    # ts.get_text_statistics(d_norm, "normData")

    # classified data
    rf_score = classifiers.get_all_classifiers_evaluations(d_norm)
    tn_rf_score = classifiers.get_all_classifiers_evaluations(d_norm)
    classifiers.compare_our_result(rf_score, tn_rf_score)

    # K-means clustering
    clustering.get_class(d_norm)

    vectors, vocab = tm.data_processing(d_norm)
    u, s, vh = tm.svd(vectors)
    print(tm.show_topics(vh[:6], vocab))


if __name__ == "__main__":
    main()

