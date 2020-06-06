import numpy as np
import pandas as pd


def read_data(file_name, path, label):
    lst = []
    with open(path + '/' + file_name, 'r', encoding="utf8") as f:
        for line in f:
            cur_dict = {"text": [], "label": []}
            cur_dict["text"] = [line]
            cur_dict["label"] = [label]
            cur_df = pd.DataFrame(cur_dict)
            lst.append(cur_df)
        total_df = pd.concat(lst)
        # return total_df
        out_path = path + file_name + ".csv"
        total_df.to_csv(out_path, index=False, encoding='utf-8')

    with open(out_path, 'rb') as data:
        return pd.read_csv(data)
