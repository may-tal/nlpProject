import textStatistics as ts
from data import *
import classifiers
import textNormalization as tn

PATH = 'Data/Data/clean'


def main():
    # d = read_data(['sentences.neg', 'sentences.pos'], PATH)

    # text normalization (get the clean data)
    # d_norm = tn.text_normalization(d)
    # d_norm.to_csv('d_norm.csv', index=False, encoding='utf-8')
    d_norm = pd.read_csv('d_norm.csv')

    # text statistics
    # ts.get_text_statistics(d, "data")
    # ts.get_text_statistics(d_norm, "normData")

    # classified data
    # rf_score = classifiers.get_all_classifiers_evaluations(d)
    # tn_rf_score = classifiers.get_all_classifiers_evaluations(d_norm)
    # classifiers.compare_our_result(rf_score, tn_rf_score)

    x = tn.get_hebrew_stopwords()
    print(x)

if __name__ == "__main__":
    main()

