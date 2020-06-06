import textStatistics as ts
from data import *

PATH = 'Data/Data/clean'

def main():
    ts.plot_top_20_common_words(read_data('sentences.neg', PATH, '1'))

if __name__ == "__main__":
    main()

