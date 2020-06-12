import textStatistics as ts
from data import *
<<<<<<< HEAD
import cleanData as cd
import classifiers
=======
import textNormalization as tn
>>>>>>> df3f9904e0aeb195e0acc8d98dd5006d5642116c

PATH = 'Data/Data/clean'

def main():
<<<<<<< HEAD
    d = read_data(['sentences.neg', 'sentences.pos'], PATH)
    #ts.plot_top_non_stop_words_barchart(d)
    # for idx, row in d.iterrows():
    #     ts.print_yap_analysis(row["text"])
    #     break
    #cd.stemmer_and_lemmatizer(d, 'out')
    train_df, test_df = classifiers.prepare_data_for_classify(d)
    #classifier.bag_of_words(train_df, test_df)
    classifiers.data_exploration(train_df)
    x_train_counts, x_test_counts = classifiers.bag_of_words(train_df, test_df)
    x_train_tf, x_test_tfidf = classifiers.tf_idf(x_train_counts, x_test_counts)
    classifiers.multinomial_naive_bayes_classifier(x_train_tf, train_df, x_test_tfidf)

=======
    d = read_data('sentences.neg', PATH, '1')
    d = tn.text_normalization(d)
    print("hi")
>>>>>>> df3f9904e0aeb195e0acc8d98dd5006d5642116c

if __name__ == "__main__":
    main()

