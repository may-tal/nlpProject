import textStatistics as ts
from data import *
import cleanData as cd

PATH = 'Data/Data/clean'

def main():
    d = read_data('sentences.neg', PATH, '1')
    #ts.plot_top_non_stop_words_barchart(d)
    # for idx, row in d.iterrows():
    #     ts.print_yap_analysis(row["text"])
    #     break
    cd.stemmer_and_lemmatizer(d, 'out')


if __name__ == "__main__":
    main()

