import re
import string

import contractions
import nltk
import numpy as np
import pandas as pd
from sklearn import model_selection

DISCUSSIONS_DATASET_PATH = "../data/IMapBook_discussions_dataset.xlsx"

COL_MESSAGE = "Message"
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


def load_annotated_discussions_data(keep_punctuation=True) -> pd.DataFrame:
    data_df = pd.read_excel(DISCUSSIONS_DATASET_PATH)
    data_df[COL_TARGET] = preprocess_target(data_df[COL_TARGET])
    data_df[COL_MESSAGE] = preprocess_messages(data_df[COL_MESSAGE], keep_punctuation)
    return data_df


def preprocess_target(target: pd.Series) -> pd.Series:
    return target.str.strip().apply(lambda val: TARGET_MAPPING.get(val, val))


def preprocess_messages(messages: pd.Series, keep_punctuation=True) -> pd.Series:
    """Since messages are short, keeping punctuation might help with classification."""
    stemmer = nltk.stem.WordNetLemmatizer()
    to_replace = [re.escape(p) for p in string.punctuation]
    replace_with = [f" {p} " for p in string.punctuation] if keep_punctuation else ""

    def _lemmatize_text(text):
        return " ".join([stemmer.lemmatize(word) for word in nltk.word_tokenize(text)])

    return (
        messages.str.lower()
        .apply(contractions.fix)
        .apply(_lemmatize_text)
        .replace(to_replace, replace_with, regex=True)
    )


def evaluate_models(models, messages, target):
    evaluations = {}
    for model in models:
        cross_val_scores = cross_validate_model(model, messages, target)
        evaluations[str(model)] = {score_name: np.mean(scores) for score_name, scores in cross_val_scores.items()}
    return evaluations


def cross_validate_model(model, messages, target, n_splits=4, random_state=1, shuffle=True):
    """Model should inherit from sklearn.base.BaseEstimator, get_params and set_params are needed for cloning."""
    kfold = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    scores = model_selection.cross_validate(
        estimator=model,
        X=messages,
        y=target,
        cv=kfold,
        scoring=("accuracy", "f1_macro", "neg_log_loss"),
    )
    return scores
