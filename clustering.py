from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from bidi.algorithm import get_display
import numpy as np
import feature_extraction


def K_means_clustering(train_df):
    """
    K-means clustering algorithm with k=3 for cluster tha messages to their violence type
    :param train_df: data to cluster
    """
    tfidf_vectorizer = TfidfVectorizer(lowercase=False)
    X = tfidf_vectorizer.fit_transform(train_df.text)

    true_k = 3
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=400, n_init=15)
    model.fit(X)
    y_kmeans = model.predict(X)

    #  print the top words per cluster
    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = tfidf_vectorizer.get_feature_names()
    for i in range(true_k):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :4]:
            print(' %s' % terms[ind]),

    print("\n")

    print("Prediction")

    Y = tfidf_vectorizer.transform(["זונה אחת"])
    prediction = model.predict(Y)
    print(prediction)

    Y = tfidf_vectorizer.transform(["ילדה בכיינית ומכוערת"])
    prediction = model.predict(Y)
    print(prediction)


    def plot_WordCloud(cluster_num):
        bidi_text = get_display(" ".join(np.array(terms)[order_centroids[cluster_num, :15]][::-1]))
        print(bidi_text)
        pos_wordcloud = WordCloud(width=600, height=400, font_path='externals/FreeSans/FreeSansBold.ttf').generate(bidi_text)
        plt.figure(figsize=(10, 8), facecolor='k')
        plt.imshow(pos_wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.show()

    for i in range(true_k):
        plot_WordCloud(i)

    # # Create K Clusters of 15
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
    x_train_tf, x_test_tfidf, train_df, test_df = feature_extraction.get_bow_tfidf(data, False)
    K_means_clustering(train_df)
