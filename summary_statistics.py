import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import table

pd.set_option('display.max_columns', None)
data = pd.read_csv('data_cleaned.csv', parse_dates = ['created_at'])
sum_statistics = pd.DataFrame({
    'Data Types': data.dtypes,
    'Unique Values': data.nunique(),
    'Missing Values': data.isnull().sum(),
    'Missing Values (%)': round(data.isnull().mean() * 100, 2)
})

print(sum_statistics)
print()
print(data.describe())
print()
print(data[['state_code', 'candidate']].value_counts())

# Plot 1
sns.displot(data = data, x = 'state_code', hue = 'candidate', kde = True, aspect = 3)
plt.xlabel('State Code')
plt.title('Distribution of Tweets by State and Candidate')
plt.show()
plt.savefig('distr_of_tweets_by_state_and_candidate.png')

# Plot 2
sns.displot(data = data, x = 'created_at', hue = 'candidate', kde = True, aspect = 3)
plt.xlabel('Date')
plt.title('Distribution of Tweets by Date and Candidate')
plt.show()
plt.savefig('distr_of_tweets_by_date_and_candidate.png')




