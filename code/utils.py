import re
import string
from typing import Tuple

import contractions
import nltk
import numpy as np
import pandas as pd
from sklearn import model_selection

DISCUSSIONS_DATASET_PATH = "../data/IMapBook_discussions_dataset.xlsx"

COL_MESSAGE = "Message"
COL_COLLAB_RESP = "Collab Response"
COL_PSEUDONYM = "Pseudonym"
COL_BOOKCLUB = "Bookclub"
COL_RESP_NUMBER = "Response Number"

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
    data_df = joined_df[[COL_MESSAGE, COL_BOOKCLUB, COL_PSEUDONYM, COL_RESP_NUMBER, COL_COLLAB_RESP]].copy()

    data_df.loc[:, COL_MESSAGE] = preprocess_text(data_df[COL_MESSAGE], keep_punctuation)
    data_df.loc[:, COL_COLLAB_RESP] = preprocess_text(data_df[COL_COLLAB_RESP], keep_punctuation)

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
