import ast
import pandas as pd
from gensim.models import Word2Vec
from my_functions import cleaning, document_vector, cosine_sim
import multiprocess
#import nltk
#from nltk.corpus import opinion_lexicon

# Reading the datasets
trump = pd.read_csv('data/hashtag_donaldtrump.csv',
                    lineterminator='\n',
                    usecols = ['created_at', 'tweet', 'country','state_code'])
biden = pd.read_csv('data/hashtag_joebiden.csv',
                    lineterminator='\n',
                    usecols = ['created_at', 'tweet', 'country','state_code'])


# Data cleaning
trump = cleaning(trump, candidate = 'Trump')
biden = cleaning(biden, candidate = 'Biden')
data = pd.concat([trump, biden], ignore_index = True) # concatenating the dataframes

# If you want to save the results before proceeding
"""data.to_csv('data_cleaned.csv', encoding='utf-8', header=True)
data = pd.read_csv('data_cleaned.csv')
data['clean_text'] = data['clean_text'].apply(ast.literal_eval)"""

model = Word2Vec(data['clean_text'],
                 vector_size = 100,
                 window = 10,
                 min_count = 1,
                 workers = multiprocess.cpu_count())
data['vector'] = data['clean_text'].apply(lambda x: document_vector(x, model))

# Download opinion lexicon (maybe will use this later)
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

# Computing the positivity vector
positive_vector = document_vector(good_words, model)
negative_vector = document_vector(bad_words, model)
positivity_vector = positive_vector - negative_vector

# Calculating the cosine similarity between positivity vector and each word vector of a tweet
data['sentiment_score'] = data['vector'].apply(lambda x: cosine_sim(x, positivity_vector))

#data['sentiment_score'] = pd.to_numeric(data['sentiment_score']) # converting to numeric data type

# Calculating the mean of sentiment scores for each date, candidate, state group
data = data.groupby(['created_at', 'candidate', 'state_code'])['sentiment_score'].mean().reset_index()


# Pivot the data to have one column per candidate
pivoted_data = (data.pivot_table(
    index = ['created_at', 'state_code'],
    columns = 'candidate',
    values = 'sentiment_score')
)

# Reading actual results
actual_results = pd.read_csv('data/actual_results_by_state.csv',
                             usecols = ['state_abr', 'state', 'trump_win', 'biden_win'])

# Merging the actual results with the above created dataframe
data = data.merge(actual_results, left_on = 'state_code', right_on = 'state_abr', how = 'left')
data = data[['created_at', 'state_code', 'Biden', 'Trump', 'state', 'trump_win', 'biden_win']]

# Creating a dataframe for overall picture (not States based mean sentiment scores for both candidates over time)
data_general = data.copy().groupby('created_at').agg({'Biden': "mean", 'Trump': 'mean'}).reset_index()
# Saving the data
data.to_csv('data_sentiments.csv', encoding='utf-8', header=True)
data.to_csv('data_general.csv', encoding='utf-8', header=True)
