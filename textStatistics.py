import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
#import stanza
import requests
import json
from time import sleep
import pandas as pd
from pandas.io.json import json_normalize

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

def plot_top_20_common_words(data):
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
    plt.show(fig)


def invert_words(words):
    """
    this function invert every word from the given list
    :param words:
    :return:
    """
    return [w[::-1] for w in words]

def plot_top_non_stop_words_barchart(data):
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

def get_hebrew_stop_words():
    stop_words = "heb_stopwords.txt"
    with open(stop_words, encoding="utf-8") as in_file:
        lines = in_file.readlines()
        res = [l.strip() for l in lines]
        return res

def print_yap_analysis(text):
    text = text.replace(r'"', r'\"')
    url = f'https://www.langndata.com/api/heb_parser?token=a70b54d01ef5e9c055ab9051b9deafee'
    _json = '{"data":"' + text.strip() + '"}'
    #         print(url)
    #         print(_json)
    headers = {'content-type': 'application/json'}
    sleep(0.5)
    r = requests.post(url, data=_json.encode('utf-8'), headers={'Content-type': 'application/json; charset=utf-8'})
    json_obj = r.json()

    md_lattice = json_obj["md_lattice"]
    res_df = pd.io.json.json_normalize([md_lattice[i] for i in md_lattice.keys()])
    return res_df
