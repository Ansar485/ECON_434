import pandas as pd
from my_functions import cleaning
import gensim

trump = pd.read_csv('data/hashtag_donaldtrump.csv', lineterminator='\n')
biden = pd.read_csv('data/hashtag_joebiden.csv', lineterminator='\n')

trump = cleaning(trump, candidate = 'Trump')
biden = cleaning(biden, candidate = 'Biden')









