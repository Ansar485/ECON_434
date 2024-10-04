# Objective:
The project's main objective is to analyze the tweeter users' sentiments related to the US 2020 presidential elections and construct a model that can accurately predict results of the election by states. This will be done by using Word2Vec from the gensim Python library. To be specific, the Continuous Bag of Words (CBOW) model will be used in order to get a deeper insight into the sentiments associated with selected political issues and actors. We will then assess its efficiency by computing the error rate between the model’s predictions and the actual results, by states.

# Data:
The data is retrieved from [Kaggle](https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets).
The brief summary of the dataset: Tweets collected, using the Twitter API **statuses_lookup** and **snsscrape** for keywords, with the original intention to try to update this dataset daily so that the timeframe will eventually cover 15.10.2020 and 04.11.2020. **Added 06.11.2020** With the events of the election still ongoing as of the date that this comment was added, I've decided to keep updating the dataset with tweets until at least the end of the 6th Nov. **Added 08.11.2020**, just one more version pending to include tweets until at the end of the 8th Nov.
