import textStatistics as ts
from data import *
import classifiers
import textNormalization as tn

PATH = 'Data/Data/clean'

def main():
    d = read_data(['sentences.neg', 'sentences.pos'], PATH)

    # text normalization (get the clean data)
    d_norm = tn.text_normalization(d)
    d_norm.to_csv('d_norm.csv', index=False, encoding='utf-8')

    # classified data
if __name__ == "__main__":
    main()

