from gensim.models import Word2Vec
from my_functions import document_vector, cosine_sim
from gensim.utils import simple_preprocess
# Gotta use VADER


# Load the model
model = Word2Vec.load("word2vec_model.model")

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

# Checking the model behavior with unknown sentences
bad_sentence = simple_preprocess('Ansar is a bad, even terrible, student, he does not attend lectures at all!')
good_sentence = simple_preprocess('Dr. Lagios is a very chill professor. We like him!')

bad_sentence_vec = document_vector(bad_sentence, model)
good_sentence_vec = document_vector(good_sentence, model)

print(f'The sentiment score of a bad sentence is: {cosine_sim(bad_sentence_vec, positivity_vector)}')
print(f'The sentiment score of a good sentence is: {cosine_sim(good_sentence_vec, positivity_vector)}')
