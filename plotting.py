import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
# Reading the data
data = pd.read_csv('data_sentiments.csv', parse_dates = ['created_at'])
data['created_at'] = data['created_at'].dt.date
actual_results = pd.read_csv('data/actual_results_by_state.csv',
                             usecols = ['state_abr', 'state', 'trump_win', 'biden_win'])
data = data.merge(actual_results, left_on = 'state_code', right_on = 'state_abr', how = 'left')
data = data[['created_at', 'state_code', 'Biden', 'Trump', 'state', 'trump_win', 'biden_win']]
pd.set_option('display.max_columns', None)

data_general = data.copy().groupby('created_at').agg({'Biden': "mean", 'Trump': 'mean'}).reset_index()
plt.figure(figsize=(10, 6))

# Plot sentiment scores for each candidate over time
for candidate in data_general.columns:
    if candidate in ['Trump', 'Biden']:
        plt.plot(data_general['created_at'], data_general[candidate], label = candidate)

# Step 4: Customize the plot
plt.title('Sentiment Scores Over Time by Candidate')
plt.xlabel('Date')
plt.ylabel('Sentiment Score')
plt.legend(title='Candidate')
plt.grid(True)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Show the plot
plt.tight_layout()
plt.show()