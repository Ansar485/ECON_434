import pandas as pd
import numpy as np
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
    df = df[['created_at', 'tweet', 'country','state_code']].dropna()  # dropping NA's
    df = df[df['country'].str.contains('United States')] # Filtering for only USA
    df['created_at'] = pd.to_datetime(df['created_at']).dt.date # Converting to date
    df = df[df['tweet'].str.len() >= 10] # Remove tweets with less than 10 characters
    df['language'] = df['tweet'].apply(detect_language)
    df = df[df['language'] == 'en'] # filtering for texts in english
    df['clean_text'] = df['tweet'].apply(gensim.utils.simple_preprocess) # text preprocessing (parallelized or vectorized if possible)
    df['candidate'] = candidate  # Creating candidate column
    df = df[['created_at', 'clean_text', 'state_code', 'candidate']].dropna() # selecting relevant columns
    return df

def document_vector(tokens, model):
    # Remove out-of-vocabulary words
    tokens = [word for word in tokens if word in model.wv]
    if len(tokens) == 0:
        return np.zeros(model.vector_size)
    return np.mean(model.wv[tokens], axis=0)

def cosine_sim(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]