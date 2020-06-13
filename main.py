import textStatistics as ts
from data import *
import classifiers
import textNormalization as tn

PATH = 'Data/Data/clean'

def main():
    d = read_data(['sentences.neg', 'sentences.pos'], PATH)
    #ts.plot_top_non_stop_words_barchart(d)
    # for idx, row in d.iterrows():
    #     ts.print_yap_analysis(row["text"])
    #     break
    #cd.stemmer_and_lemmatizer(d, 'out')

    # train_df, test_df = classifiers.prepare_data_for_classify(d)
    # #classifier.bag_of_words(train_df, test_df)
    # classifiers.data_exploration(train_df)
    # x_train_counts, x_test_counts = classifiers.bag_of_words(train_df, test_df)
    # x_train_tf, x_test_tfidf = classifiers.tf_idf(x_train_counts, x_test_counts)
    # classifiers.multinomial_naive_bayes_classifier(x_train_tf, train_df, x_test_tfidf)

    d_norm = tn.text_normalization(d)
    d_norm.to_csv('d_norm.csv', index=False, encoding='utf-8')

if __name__ == "__main__":
    main()

