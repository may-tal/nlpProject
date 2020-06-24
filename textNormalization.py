import textStatistics as ts
import pandas as pd
import re

PATH = 'Data/Data/clean'
SLANG_DICT = {"אחשלי": "אח שלי", "אמשך": "אמא שלך", "אבשך": "אבא שלך", "חיימשלי": "חיים שלי", "אמאשלך": "אמא שלך",
              "אבאשלך": "אבא שלך", "כוסאמק": "כוס אמק", "כוסאמאשלך": "כוס אמא שלך", "כוסאבאשלך": "כוס אבא שלך",
              "אינבעיה": "אין בעיה", "בנזונה": "בן זונה", "בתזונה": "בת זונה", "החיימשלי": "החיים שלי",
              "היפשלי": "היפה שלי", "בצפר": "בית ספר", "כוסעמקערס": "כוס עמק ערס", "אנלא": "אני לא",
              "אנילא": "אני לא", "אמאשך": "אמא שלך", "אבאשך": "אבא שלך"}


def text_normalization(data):
    data = remove_duplicates_characters(data)
    data = translate_slang(data)
    data = stemmer_and_lemmatizer(data)
    data = get_text_non_stopwords(data)
    return data


def stemmer_and_lemmatizer(data):
    for idx, row in data.iterrows():
        yap_df = ts.get_yap_analysis(row["text"])
        lemme = yap_df["lemma"].values
        new_row = ' '.join(lemme)
        new_row += "\n"
        data.iloc[[idx], [0]] = new_row
    return data


def remove_duplicates_characters(data):
    """
    this function remove duplicates characters in words
    """
    for idx, row in data.iterrows():
        new_row = ""
        words = row["text"].split()
        for word in words:
            if all(c in "פחעהא" for c in word):
                no_dup_word = word
            elif "כוס" in word and "אמא" in word and "שלך" in word:
                no_dup_word = "כוס אמא שלך"
            elif "כוס" in word and "אבא" in word and "שלך" in word:
                no_dup_word = "כוס אבא שלך"
            elif "אמא" in word and ("שך" in word or "שלך" in word):
                no_dup_word = "אמאשך"
            elif "אבא" in word and ("שך" in word or "שלך" in word):
                no_dup_word = "אבאשך"
            else:
                no_dup_word = re.sub(r'(.)\1{2,}', r'\1', word)
                if len(no_dup_word) >= 2:
                    if no_dup_word[-2] == no_dup_word[-1]:
                        if no_dup_word[-1] in {'ך', 'ן', 'ף'}:
                            no_dup_word = no_dup_word[:-1]
                    elif no_dup_word[0] == no_dup_word[1] and no_dup_word[0] == 'י':
                        no_dup_word = no_dup_word[1:]
                    if "כוס" in no_dup_word and no_dup_word[-1] == no_dup_word[-2]:
                        no_dup_word = no_dup_word[:-1]
                    if "אח" in no_dup_word and no_dup_word[-1] == no_dup_word[-2]:
                        no_dup_word = no_dup_word[:-1]
                if re.match(r'ז+ו+נ+ה+', no_dup_word):
                    no_dup_word = "זונה"
                if re.match(r'י+א+ל+ה+', no_dup_word):
                    no_dup_word = "יאלה"
                if re.match(r'א+ו+ת+ך+', no_dup_word):
                    no_dup_word = "אותך"
                if re.match(r'ב+נ+ז+ו+נ+ה+', no_dup_word):
                    no_dup_word = "בנזונה"
                if re.match(r'ב+ת+ז+ו+נ+ה+', no_dup_word):
                    no_dup_word = "בנזונה"
            no_dup_word = re.sub("\\?+", "?", word)
            no_dup_word = re.sub("\\.+", "?", word)
            new_row += no_dup_word + " "
        new_row = new_row[:-1]
        new_row += "\n"
        data.iloc[[idx], [0]] = new_row
    return data


def translate_slang(data):
    """
    this function translate slang to hebrew word
    """
    for idx, row in data.iterrows():
        new_row = ""
        words = row["text"].split()
        for word in words:
            if word in SLANG_DICT.keys():
                new_row += SLANG_DICT[word] + " "
            else:
                new_row += word + " "
        new_row = new_row[:-1]
        new_row += "\n"
        data.iloc[[idx], [0]] = new_row
    return data


def fix_yap(data):
    """
    This function fix the data after yap
    """
    for idx, row in data.iterrows():
        new_row = ""
        words = row["text"].split()
        for word in words:
            if word == "מך":
                new_row += "מכות" + " "
            elif word == "יבן":
                new_row += "יא בן" + " "
            elif word == "זנה":
                new_row += "זונה" + " "
            elif word == "יבת":
                new_row += "יא בת" + " "
            else:
                new_row += word + " "
        new_row = new_row[:-1]
        new_row += "\n"
        data.iloc[[idx], [0]] = new_row
    return data


def get_text_non_stopwords(data):
    """
    This function return the text without stop words
    """
    heb_stopwords = get_hebrew_stopwords()
    for idx, row in data.iterrows():
        words = row['text'].split()
        new_row = ""
        for word in words:
            if word in heb_stopwords:
                continue
            else:
                new_row += word + " "
        new_row = new_row[:-1]
        data.iloc[[idx], [0]] = new_row
    return data


def get_hebrew_stopwords():
    """
    This function return the hebrew stopwords as list
    """
    with open('new_heb_stopwords.txt', encoding="utf-8") as in_file:
        lines = in_file.readlines()
        res = [l.strip() for l in lines]
    res.extend([",", "."])
    return res