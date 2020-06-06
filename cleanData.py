import textStatistics as ts
import pandas as pd

PATH = 'Data/Data/clean'

def stemmer_and_lemmatizer(data, file_name):
    lst = []
    for idx, row in data.iterrows():
        cur_dict = {"text": [], "label": []}
        print(row["text"])
        yap_df = ts.print_yap_analysis(row["text"])
        lemme = yap_df["lemma"].values
        new_row = ' '.join(lemme)
        cur_dict["text"] = [new_row]
        cur_dict["label"] = [row["label"]]
        cur_df = pd.DataFrame(cur_dict)
        lst.append(cur_df)
    total_df = pd.concat(lst)
    out_path = PATH + file_name + ".csv"
    total_df.to_csv(out_path, index=False, encoding='utf-8')



