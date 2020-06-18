import textStatistics as ts
from data import *
import classifiers as clf
import textNormalization as tn
import clustering

PATH = 'Data/Data/clean'

def main():
    #d = read_data(['sentences.neg', 'sentences.pos'], PATH)

    # text normalization (get the clean data)
    # d_norm = tn.text_normalization(d)
    # d_norm.to_csv('d_norm.csv', index=False, encoding='utf-8')
    d_norm = pd.read_csv('d_norm.csv')

    # text statistics
    # ts.get_text_statistics(d, "data")
    # ts.get_text_statistics(d_norm, "normData")

    # classified data
    #clf.get_all_classifiers_evaluations(d)
    clustering.get_calss(d_norm)


if __name__ == "__main__":
    main()

