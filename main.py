import textStatistics as ts
from data import *
import textNormalization as tn

PATH = 'Data/Data/clean'

def main():
    d = read_data('sentences.neg', PATH, '1')
    #ts.plot_top_non_stop_words_barchart(d)
    # for idx, row in d.iterrows():
    #     ts.print_yap_analysis(row["text"])
    #     break
    # tn.stemmer_and_lemmatizer(d, 'out')
    tn.remove_duplicates_charcters(d)

if __name__ == "__main__":
    main()

