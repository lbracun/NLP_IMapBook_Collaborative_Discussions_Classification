import re
import string
from typing import Tuple

import contractions
import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from sklearn import model_selection

DISCUSSIONS_DATASET_PATH = "../data/IMapBook_discussions_dataset.xlsx"

COL_MESSAGE = "Message"
COL_COLLAB_RESP = "Collab Response"
COL_PSEUDONYM = "Pseudonym"
COL_BOOKCLUB = "Bookclub"
COL_RESP_NUMBER = "Response Number"
COL_BOOKID = "Book ID"
COL_MESSAGE_LEN = "Message Length"
COL_BOOK_SIMILARITY = "Book Similarity"
COL_CONTAINS_LINK = "Contains Link"
COL_CONTAINS_EMOTICON = "Contains Emoticon"
COL_UPPERCASE_NUM = "Number of uppercase letters"
COL_QUESTION_NUM = "Number of question words"

COL_TARGET = "CodePreliminary"

TARGET_MAPPING = {
    "Instuction Question": "Assignment Instructions",
    "Instruction Question": "Assignment Instructions",
    "Opening statement": "Opening Statement",
    "Incomplete/typo": "Incomplete/Typo",
    "General Comment (Narrative?)": "General Comment",
    "External Material": "Outside Material",
    "Assignment Instructions Question": "Assignment Instructions",
    "Content Discussion/Outside Material": "Content Discussion",
    "Non-verbal": "Emoticon/Non-verbal",
    "Observation": "Feedback",
    "General Question": "General Discussion",
}


def load_discussions_data(keep_punctuation=True) -> Tuple[pd.DataFrame, pd.Series]:
    xlsx = pd.ExcelFile(DISCUSSIONS_DATASET_PATH)

    df1 = pd.read_excel(xlsx, sheet_name=0)
    df2 = pd.read_excel(xlsx, sheet_name=1)
    resp_df = pd.read_excel(xlsx, sheet_name=2, index_col=COL_RESP_NUMBER)

    joined_df = df1.append(df2).join(resp_df, on=COL_RESP_NUMBER)

    target_df = preprocess_target(joined_df[COL_TARGET])
    data_df = joined_df[[COL_MESSAGE, COL_BOOKID, COL_BOOKCLUB, COL_PSEUDONYM, COL_RESP_NUMBER, COL_COLLAB_RESP]].copy()

    data_df[COL_CONTAINS_EMOTICON] = data_df.apply(emoticons_feature, axis=1)
    data_df[COL_CONTAINS_LINK] = data_df.apply(has_link_feature, axis=1)
    data_df[COL_UPPERCASE_NUM] = data_df.apply(uppercase_letters_feature, axis=1)
    data_df.loc[:, COL_MESSAGE] = preprocess_text(data_df[COL_MESSAGE], keep_punctuation)
    data_df.loc[:, COL_COLLAB_RESP] = preprocess_text(data_df[COL_COLLAB_RESP], keep_punctuation)
    data_df[COL_BOOK_SIMILARITY] = data_df.apply(message_book_similarity, axis=1)
    data_df[COL_QUESTION_NUM] = data_df.apply(questions_feature, axis=1)

    print(data_df)

    return data_df, target_df


def preprocess_target(target: pd.Series) -> pd.Series:
    return target.str.strip().apply(lambda val: TARGET_MAPPING.get(val, val))


def preprocess_text(text_col: pd.Series, keep_punctuation=True) -> pd.Series:
    """Since messages are short, keeping punctuation might help with classification."""
    stemmer = nltk.stem.WordNetLemmatizer()
    to_replace = [re.escape(p) for p in string.punctuation]
    replace_with = [f" {p} " for p in string.punctuation] if keep_punctuation else ""

    def _lemmatize_text(text):
        return " ".join([stemmer.lemmatize(word) for word in nltk.word_tokenize(text)])

    return (
        text_col.fillna("")
        .str.lower()
        .apply(contractions.fix)
        .apply(_lemmatize_text)
        .replace(to_replace, replace_with, regex=True)
    )


def evaluate_models(models, data_df, target_df):
    evaluations = {}
    for model in models:
        cross_val_scores = cross_validate_model(model, data_df, target_df)
        evaluations[str(model)] = {score_name: np.mean(scores) for score_name, scores in cross_val_scores.items()}
    return evaluations


def cross_validate_model(model, data_df, target_df, n_splits=4, random_state=1, shuffle=True):
    """Model should inherit from sklearn.base.BaseEstimator, get_params and set_params are needed for cloning."""
    kfold = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    scores = model_selection.cross_validate(
        estimator=model,
        X=data_df,
        y=target_df,
        cv=kfold,
        scoring=("accuracy", "f1_macro", "neg_log_loss"),
    )
    return scores

def preprocess_book(title: string):
    path = "../data/books/" + title

    with open(path, 'r') as book: 
        book = book.read().replace('“', '"').replace('”', '"').replace("’", "'").replace("—", "-").replace("…", "")
        text = book.lower()        

        table = text.maketrans({key: "" for key in string.punctuation})
        text = text.translate(table)

        lemmatizer = nltk.stem.WordNetLemmatizer()

        tokens = []
        for token in nltk.word_tokenize(text): 
            tokens.append(lemmatizer.lemmatize(token))

        tokens = [token for token in tokens if token not in stopwords.words("english")]
        freq = FreqDist(tokens)
        #freq.plot(30)

        tokens_filtered = []
        for token in tokens: 
            if token not in tokens_filtered:
                tokens_filtered.append(token)
        
        return tokens_filtered


book1 = preprocess_book("Design_for_the_Future_When_the_Future_Is_Bleak.txt")
book2 = preprocess_book("Just_Have_Less.txt")
book3 = preprocess_book("The_Lady_or_the_Tiger.txt")

def message_book_similarity(row): 
    message = row[COL_MESSAGE].split(" ")
    bookid = row [COL_BOOKID]
    book = None
    
    if (bookid == 260 or bookid == 261):
        book = book3
    elif (bookid == 266 or bookid == 267): 
        book = book1
    else:
        book = book2

    return sum([m in book for m in message])

def word_count_feature(row):
    return len(row[COL_MESSAGE].split(" "))

def char_count_feature(row):
    return len(row[COL_MESSAGE])

def has_link_feature(row):
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    return 1 if re.match(regex, row[COL_MESSAGE]) else 0

def emoticons_feature(row):
    regex = r"(\:\w+\:|\<[\/\\]?3|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)"
    return 1 if re.match(regex, row[COL_MESSAGE]) else 0

def uppercase_letters_feature(row):
    return sum(1 for c in row[COL_MESSAGE] if c.isupper())

def questions_feature(row):
    words = ["what", "when", "who", "where", "why", "which", "how"]

    count = 0
    for word in row[COL_MESSAGE].split(" "):
        if word in words: 
            count += 1

    return count