from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def data_processing(data):
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(data.text)
