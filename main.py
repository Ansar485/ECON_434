import pandas as pd
from gensim.models import Word2Vec
from my_functions import cleaning, document_vector, cosine_sim
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import nltk
#from nltk.corpus import opinion_lexicon

# Reading the datasets
trump = pd.read_csv('data/hashtag_donaldtrump.csv', lineterminator='\n')
biden = pd.read_csv('data/hashtag_joebiden.csv', lineterminator='\n')

# Data cleaning
trump = cleaning(trump, candidate = 'Trump')
biden = cleaning(biden, candidate = 'Biden')
data = pd.concat([trump, biden], ignore_index = True) # concatenating the dataframes

#Saving the data
data.to_csv('data_cleaned.csv', encoding='utf-8', index=False, header=True)


model = Word2Vec(data['clean_text'], vector_size=100, window=10, min_count=1, workers=4)
data['vector'] = data['clean_text'].apply(lambda x: document_vector(x, model))

# Download opinion lexicon
#nltk.download('opinion_lexicon')

# Get the good (positive) and bad (negative) words
good_words = [
    "happy", "joy", "love", "excellent", "great", "wonderful", "amazing", "fantastic",
    "positive", "smile", "pleasure", "brilliant", "awesome", "bliss", "success",
    "win", "grateful", "blessed", "hope", "delight", "enthusiastic", "satisfied",
    "accomplished", "cheerful", "optimistic"
]
bad_words = [
    "sad", "hate", "terrible", "awful", "disgust", "bad", "horrible", "negative",
    "pain", "anger", "failure", "fear", "disappointment", "frustration", "upset",
    "depressed", "miserable", "anxious", "loss", "cry", "problem", "regret",
    "tragic", "worse", "guilt"
]

# Calculate the positive and negative vectors
positive_vector = document_vector(good_words, model)
negative_vector = document_vector(bad_words, model)

# Compute the positivity vector
positivity_vector = positive_vector - negative_vector


# Calculating the cosine similarity between positivity vector and each word vector of a tweet
data['sentiment_score'] = data['vector'].apply(lambda x: cosine_sim(x, positivity_vector))

print(data['sentiment_score'].mean())

# Saving the data with sentiment scores
data.to_csv('data_sentiments.csv', encoding='utf-8', index=False, header=True)

