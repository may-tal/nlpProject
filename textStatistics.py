import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import requests
from time import sleep
import pandas as pd
from pandas.io.json import json_normalize
from sklearn.feature_extraction.text import CountVectorizer


def split_by_whitespace(data):
    """
    this function split the data by whitespace
    """
    return data.split()


def count_word_number(text):
    """
    this function return the number of words in the data
    """
    return len(split_by_whitespace(text))


def plot_word_number_histogram(data):
    """
    this function plot a histogram of the number of words in the data
    """
    hist = dict()
    for idx, row in data.iterrows():
        cur_len = str(count_word_number(row['text']))
        if cur_len in hist.keys():
            hist[cur_len] += 1
        else:
            hist[cur_len] = 1
    new_hist = dict(sorted(hist.items(), key = lambda x:x[0]))
    plt.bar(new_hist.keys(), new_hist.values())
    plt.title("Word Number Histogram")
    plt.xlabel("words")
    plt.ylabel("frequancy")
    plt.show()


def plot_top_20_common_words(data, file_name):
    """
    this function plot the top 20 common words in the data
    """
    general_counter = Counter()
    for idx, row in data.iterrows():
        general_counter += Counter(split_by_whitespace(row['text']))
    most = general_counter.most_common(20)
    x, y = [], []
    for word, count in most:
            x.append(word)
            y.append(count)
    fig = sns.barplot(x=y, y=invert_words(x))
    plt.title("Top commom words")
    plt.ylabel("words")
    plt.xlabel("frequancy")
    plt.show()
    plt.savefig("statistic/" + file_name)


def plot_top_20_bigrams_words(data, file_name):
    """
    This function plot the 20 common bigrams in the data
    """
    vec = CountVectorizer(ngram_range = (2,2))
    bow = vec.fit_transform(list(data['text']))
    sum_words = bow.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)[:20]
    x, y = map(list, zip(*words_freq))
    fig = sns.barplot(x=y, y=invert_words(x))
    plt.title("Top Bigrams Barchart")
    plt.ylabel("words")
    plt.xlabel("frequancy")
    plt.show()
    plt.savefig("statistic/" + file_name)


def invert_words(words):
    """
    this function invert every word from the given list
    """
    return [w[::-1] for w in words]


def plot_top_non_stop_words_barchart(data, file_name):
    """
    this function plot toe non stop words barchart
    """
    stop = get_hebrew_stop_words()
    general_counter = Counter()
    for idx, row in data.iterrows():
        non_stop_words_text = [word for word in row['text'].split() if word not in stop]
        general_counter += Counter(non_stop_words_text)
    most = general_counter.most_common(20)
    x, y = [], []
    for word, count in most:
            x.append(word)
            y.append(count)
    fig = sns.barplot(x=y, y=invert_words(x))
    plt.title("Top Non-Stopwords Barchart")
    plt.ylabel("words")
    plt.xlabel("frequancy")
    plt.show()
    plt.savefig("statistic/" + file_name)


def get_hebrew_stop_words():
    """
    this function return the hebrew stop words
    """
    stop_words = "heb_stopwords.txt"
    with open(stop_words, encoding="utf-8") as in_file:
        lines = in_file.readlines()
        res = [l.strip() for l in lines]
        return res

def plot_top_bigrams_non_stop_words_barchart(data, file_name):
    """
    This function plot the 20 common bigrams non stop words
    """
    vec = CountVectorizer(ngram_range = (2,2))
    bow = vec.fit_transform(list(data['text']))
    sum_words = bow.sum(axis=0)
    stop = get_hebrew_stop_words()
    words_freq = []
    for word, idx in vec.vocabulary_.items():
        flag = False
        for w in word.split():
            if w in stop:
                flag = True
        if not flag:
            words_freq.append((word, sum_words[0, idx]))
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)[:20]
    x, y = map(list, zip(*words_freq))
    fig = sns.barplot(x=y, y=invert_words(x))
    plt.title("Top bigrams non stop barchart")
    plt.ylabel("words")
    plt.xlabel("frequancy")
    plt.show()
    plt.savefig("statistic/" + file_name)


def get_yap_analysis(text):
    """
    this function return yap analysis
    """
    text = text.replace(r'"', r'\"')
    url = f'https://www.langndata.com/api/heb_parser?token=a70b54d01ef5e9c055ab9051b9deafee'
    _json = '{"data":"' + text.strip() + '"}'
    sleep(3)
    r = requests.post(url, data=_json.encode('utf-8'), headers={'Content-type': 'application/json; charset=utf-8'},
                      verify=False)
    json_obj = r.json()
    md_lattice = json_obj["md_lattice"]
    res_df = pd.io.json.json_normalize([md_lattice[i] for i in md_lattice.keys()])
    print(res_df)
    return res_df


def get_text_statistics(text, text_name):
    plot_top_20_common_words(text, "commonWords_" + text_name)
    plot_top_non_stop_words_barchart(text, "commonNonStopwords_" + text_name)
    plot_top_20_bigrams_words(text, "commonBigrams_" + text_name)
    plot_top_bigrams_non_stop_words_barchart(text, "commonBigramsNonStopwords_" + text_name)

