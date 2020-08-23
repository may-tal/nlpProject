import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_data(files_name, path):
    """
    read the data files (pos and neg) to one file with binary labels when positive is 0 and negative is 1.
    The function also plot data's pie graph.
    :param files_name: data file's name
    :param path: path to tne data directory
    :return:
    """
    lst = []
    len_pos = 0
    len_neg = 0
    for file in files_name:
        with open(path + '/' + file, 'r', encoding="utf8") as f:
            for line in f:
                cur_dict = {"text": [], "label": []}
                cur_dict["text"] = [line]
                if file == 'sentences.neg':
                    cur_dict["label"] = [1]
                    len_neg += 1
                else:
                    cur_dict["label"] = [0]
                    len_pos += 1
                cur_df = pd.DataFrame(cur_dict)
                lst.append(cur_df)
    total_df = pd.concat(lst)
    out_path = path + "data.csv"
    total_df.to_csv(out_path, index=False, encoding='utf-8')

    def func(pct, data):
        a = int(pct/100.*np.sum(data))
        return "{:.1f}%\n({:d} samples)".format(pct,a)

    labels = ['Positive data', 'Negative data']
    fig1, ax1 = plt.subplots()
    ax1.pie([len_pos, len_neg],  labels=labels,  autopct=lambda pct:func(pct,[len_pos, len_neg]), startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()
    plt.savefig('data_pie.png', tranparent=True)

    with open(out_path, 'rb') as data:
        return pd.read_csv(data)
