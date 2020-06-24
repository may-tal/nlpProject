from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import classifiers as clf
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from bidi.algorithm import get_display
import numpy as np


def K_means_clustering(train_df):
    tfidf_vectorizer = TfidfVectorizer(lowercase=False)
    X = tfidf_vectorizer.fit_transform(train_df.text)

    true_k = 5
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=300, n_init=10)
    model.fit(X)
    y_kmeans = model.predict(X)

    #  print the top words per cluster
    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = tfidf_vectorizer.get_feature_names()
    for i in range(true_k):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :5]:
            print(' %s' % terms[ind]),

    print("\n")

    print("Prediction")

    Y = tfidf_vectorizer.transform(["זונה אחת"])
    prediction = model.predict(Y)
    print(prediction)

    Y = tfidf_vectorizer.transform(["ילדה בכיינית ומכוערת"])
    prediction = model.predict(Y)
    print(prediction)


    bidi_text = get_display(" ".join(np.array(terms)[order_centroids[1, :5]][::-1]))
    print(bidi_text)
    pos_wordcloud = WordCloud(width=600, height=400, font_path='externals/FreeSans/FreeSansBold.ttf').generate(bidi_text)
    plt.figure(figsize=(10, 8), facecolor='k')
    plt.imshow(pos_wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    # Create K Clusters of 15
    # k = range(1, 15)
    # # Instantiate and Fit KMeans of Clusters 1-15
    # kmeans = [KMeans(n_clusters=i) for i in k]
    # score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
    # # Plot the Elbow Method
    # plt.plot(k,score)
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Score')
    # plt.title('Elbow Curve')
    # plt.show()

def get_class(data):
    train_df, test_df = clf.prepare_data_for_classify(data)
    K_means_clustering(train_df)
