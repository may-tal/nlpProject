import textStatistics as ts
from data import *
import textNormalization as tn

PATH = 'Data/Data/clean'

def main():
    d = read_data('sentences.neg', PATH, '1')
    d = tn.text_normalization(d)
    print("hi")

if __name__ == "__main__":
    main()

