The project contains the following files:

main.py - Main project file.

classifiers.py - This file gets a tagged set of the data, split it to train and test. 
then, the classifier learns how to classify from the training set and predict tags to the test set.
This file contains few classifiers and return the scores of each one.

evaluation.py - Compute all measure scores for given classifier and plot the roc curve

clustring.py-  This file cluster the data to three class using k-means algorithm and plot wordCloud graph

data.py- this file gets folder path that contain the data files and return the data as csv form

feature_extraction.py- Transforming raw data into features that better represent the underlying problem, resulting in improved predictive model accuracy on unseen data (feature engineering process)

feature_selection.py-  This file receives train data and selects features in three methods of the feature selection algorithm - removing features with low variance, selecting the best features based on univariate statistical tests, and select from models' method.

text_statistics.py- this file contains functions that get the data after preprocessing and return the statistic of the data

text_normalization.py- this file contains all the functions of the preprocessing step - clean and normalization the data

topic_modeling.py- this file use several methods of topic modeling – SVD, NFS.

heb_stopwords.txt- text file that contain the Hebrew stop words

Data/sentences.neg- contain the data whose labelling is negative

Data/sentences.pos- contain the data whose labelling is positive

sentiment_lexicon/negative_words_he.txt – Hebrew semantic lexicon for negative words

sentiment_lexicon/positive_words_he.txt – Hebrew semantic lexicon for positive words

words_veftor.npy – pretrained Hebrew word2vec model 
				  
to run this code, all you need to do is install the relevant packages and run 'main.py' 
file.
