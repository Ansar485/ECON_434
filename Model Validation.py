from gensim.models import Word2Vec
from my_functions import document_vector, cosine_sim
from gensim.utils import simple_preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pandas as pd
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


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

# Load the model
model = Word2Vec.load("word2vec_model.model")

# Computing the positivity vector
positive_vector = document_vector(good_words, model)
negative_vector = document_vector(bad_words, model)
positivity_vector = positive_vector - negative_vector


# Checking the model behavior with unknown sentences
bad_sentence = simple_preprocess('Ansar is a bad, even a terrible, student, he does not attend lectures at all!')
good_sentence = simple_preprocess('Dr. Lagios is a very chill professor. We like him!')

bad_sentence_vec = document_vector(bad_sentence, model)
good_sentence_vec = document_vector(good_sentence, model)

print(f'The sentiment score of a bad sentence is: {cosine_sim(bad_sentence_vec, positivity_vector)}')
print(f'The sentiment score of a good sentence is: {cosine_sim(good_sentence_vec, positivity_vector)}')


# Validation through logistic regression

# Prepare data for logistic regression
data_by_states = pd.read_csv('data_by_states.csv')

# Drop rows with missing values in X and align y accordingly
X = data_by_states[['Biden', 'Trump']]
y = (data_by_states['won_actual'] == 'Biden').astype(int)  # 1 for Biden, 0 for Trump
mask = X.notna().all(axis = 1)  # Create a mask for rows without NaN values
X = X[mask]  # Filter X
y = y[mask]  # Filter y to match X

# Train logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

# Evaluate performance using AUC-ROC Score
print("\nAUC-ROC Score:", roc_auc_score(y, y_prob))


# Calculating Error Rate
total_predictions = len(data_by_states['prediction'])
correct_predictions = data_by_states['prediction'].value_counts().get('Correct', 0)
accuracy_rate = correct_predictions / total_predictions
print(f'The accuracy of the model is: {round(accuracy_rate * 100, 2)}%')

