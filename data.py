import numpy as np
import pandas as pd


def read_data(files_name, path):
    lst = []
    for file in files_name:
        with open(path + '/' + file, 'r', encoding="utf8") as f:
            for line in f:
                cur_dict = {"text": [], "label": []}
                cur_dict["text"] = [line]
                if file == 'sentences.neg':
                    cur_dict["label"] = [1]
                else:
                    cur_dict["label"] = [0]
                cur_df = pd.DataFrame(cur_dict)
                lst.append(cur_df)
    total_df = pd.concat(lst)
    # return total_df
    out_path = path + "data.csv"
    total_df.to_csv(out_path, index=False, encoding='utf-8')

    with open(out_path, 'rb') as data:
        return pd.read_csv(data)

