from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import classifiers as clf

def K_means_clustering(train_df):
    vectorizer = TfidfVectorizer(lowercase=False)
    X = vectorizer.fit_transform(train_df.text)

    true_k = 2
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=200, n_init=7)
    model.fit(X)

    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind]),

    print("\n")
    print("Prediction")

    Y = vectorizer.transform(["זונה אחת"])
    prediction = model.predict(Y)
    print(prediction)

    Y = vectorizer.transform(["ילדה בכיינית ומכוערת"])
    prediction = model.predict(Y)
    print(prediction)

def get_class(data):
    train_df, test_df = clf.prepare_data_for_classify(data)
    K_means_clustering(train_df)
