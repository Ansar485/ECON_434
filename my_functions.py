import pandas as pd
import numpy as np
import gensim
from langdetect import detect
import gensim
from langdetect.lang_detect_exception import LangDetectException
from sklearn.metrics.pairwise import cosine_similarity

def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def cleaning(df, candidate):
    df = df[(df['country'] == 'United States of America') | (df['country'] == 'United States')] # only USA
    df = df[['created_at', 'tweet', 'state_code']] # selecting only relevant columns
    df = df.dropna()  # removing all the na's
    df['candidate'] = candidate # creating candidate column
    df['created_at'] = pd.to_datetime(df['created_at']) # formatting as pd datetime
    df = df[(df['tweet'].str.len() >= 10)] # remove meaningless tweets (tweets that are less than 10 characters)
    # didn't remove replies, maybe I should, gotta think about it
    df = df[df['tweet'].apply(detect_language) == 'en'] # filtering only for texts in english
    df['clean_text'] = df.tweet.apply(gensim.utils.simple_preprocess) # text preprocessing
    df = df[['created_at', 'clean_text', 'state_code', 'candidate']]
    return df

def document_vector(tokens, model):
    # Remove out-of-vocabulary words
    tokens = [word for word in tokens if word in model.wv]
    if len(tokens) == 0:
        return np.zeros(model.vector_size)
    return np.mean(model.wv[tokens], axis=0)


def cosine_sim(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]