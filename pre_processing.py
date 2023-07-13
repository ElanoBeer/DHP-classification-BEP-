# ------------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------------

import pandas as pd

import gensim
from gensim.parsing.preprocessing import stem_text
from spacy.lang.en import STOP_WORDS

STOP_WORDS = list(STOP_WORDS)

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download the WordNet corpus
nltk.download("wordnet")
nltk.download("omw-1.4")

# ------------------------------------------------------------------------------------------------------
# Creating 3 different pre-processing methods
# ------------------------------------------------------------------------------------------------------

df = pd.read_excel("../Datasets/HealthPlatforms_BEP.xlsx")
labels = pd.read_excel("../Datasets/final_sample_Elano.xlsx")


def remove_stopwords(text: list, stop_words=STOP_WORDS):
    """Removes stop words in a text column evaluated per token.

    Args:
        text (list) : a list of tokenized string word
        stop_words (list): Defaults to STOP_WORDS.

    Returns:
        no_stopwords_text (list): a list of tokenized strings without stop words.
    """
    no_stopwords_text = [token for token in text if token not in stop_words]
    return no_stopwords_text


def preprocess(df: pd.DataFrame, labels: pd.DataFrame, method: str):
    """A function that pre-processes the data based on the pre-defined baseline settings.

    Args:
        data (pd.DataFrame): The base dataframe with company data.
        labels (pd.DataFrame): The dataframe that contains the social platform labels.

    Returns:
        text_df: The dataframe that contains a tokenized and lowercased text column as well as the labels.
    """

    # remove the empty instances inplace
    df.dropna(how="all", axis=1, inplace=True)

    # create the text feature
    text_df = df.copy()
    text_df = text_df[["id", "NAME", "TAGLINE", "Full Description"]]
    text_df["Full Description"].fillna(text_df["TAGLINE"], inplace=True)
    text_df.rename(columns={"Full Description": "text"}, inplace=True)

    # add the label based on the id
    label_dict = labels.set_index("id")["Elano"].to_dict()
    text_df = text_df.assign(label=text_df["id"].map(label_dict))

    # Clean data using the built in cleaner in gensim
    text_df = text_df[text_df["text"].notna()]

    if method == "stemming":
        stemmer = PorterStemmer()
        text_df["text_clean"] = (
            text_df["text"]
            .apply(lambda x: gensim.utils.simple_preprocess(x))
            .apply(lambda x: remove_stopwords(x))
            .apply(lambda x: [stemmer.stem(word) for word in x])
        )
        return text_df

    if method == "lemmatization":
        lemmatizer = WordNetLemmatizer()
        text_df["text_clean"] = (
            text_df["text"]
            .apply(lambda x: gensim.utils.simple_preprocess(x))
            .apply(remove_stopwords)
            .apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
        )
        return text_df

    text_df["text_clean"] = (
        text_df["text"]
        .apply(lambda x: gensim.utils.simple_preprocess(x))
        .apply(lambda x: remove_stopwords(x))
    )

    return text_df


# ------------------------------------------------------------------------------------------------------
# Obtaining the labeled and unlabeled datasets
# ------------------------------------------------------------------------------------------------------


def label_split(df):
    """Split the pre-processed dataset in a labeled and unlabeled set."""

    labeled_df = df[(df["label"].notna()) & (df["text"].notna())]
    unlabeled_df = df[(df["label"].isna()) & (df["text"].notna())]

    return labeled_df, unlabeled_df
