import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly
from pandas import read_csv
import seaborn as sns
import os
pd.set_option('display.max_columns', None)

# Reading the data
data_sentiments = pd.read_csv('data_sentiments.csv', parse_dates = ['created_at'])
data_sentiments['created_at'] = data_sentiments['created_at'].dt.date

data_general = read_csv('data_general.csv', parse_dates = ['created_at'])
data_general['created_at'] = data_general['created_at'].dt.date

data_by_states = read_csv('data_by_states.csv')


#Plotting density plots for Trump and Biden
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
fig.suptitle("Density Plot of Sentiment Scores")

# Plot density for Trump
sns.kdeplot(data_sentiments.Trump, ax = ax, alpha = 0.2, fill = True, color = 'red', label = 'Trump')

# Plot density for Biden
sns.kdeplot(data_sentiments.Biden, ax = ax, alpha = 0.2, fill = True, color = 'blue', label = 'Biden')

# Plotting labels and legends
ax.set_xlabel('Sentiment Score')
ax.set_ylabel('Density')
ax.legend()

plt.tight_layout()

# Creating folder 'images' to save all the images into
if not os.path.exists("images"):
    os.makedirs("images")

plt.savefig('images/Density Plots.png')
plt.show()

#Plotting sentiment scores by candidates over time
plt.figure(figsize = (10, 6))

# Plotting sentiment scores for each candidate over time
plt.plot(data_general['created_at'], data_general['Biden'], label = 'Biden', color = 'blue')
plt.plot(data_general['created_at'], data_general['Trump'], label = 'Trump', color = 'red')
# Customizing the plot
plt.title('Sentiment Scores Over Time by Candidate')
plt.xlabel('Date')
plt.ylabel('Sentiment Score')
plt.legend(title = 'Candidate')
plt.grid(True)
plt.xticks(rotation = 45)  # rotating x-axis labels for better readability

# Showing the plot
plt.tight_layout()
plt.savefig('images/Sentiment Scores by Candidate Over Time.png')
plt.show()

# Plotting Sentiment Scores by States for Trump and Biden
fig_trump = px.choropleth(data_by_states,
                    locations = 'state_code',
                    locationmode = 'USA-states',
                    scope = 'usa',
                    color = 'Trump', labels = {'Trump': 'Sentiment score'},
                    range_color = (-1, 1),
                    hover_name = 'state_code',
                    hover_data = ['Trump'],
                    color_continuous_scale = 'reds',
                    title = 'US 2020 Elections: Trump Sentiment Scores by State'
)

plotly.offline.plot(fig_trump,
                    filename = 'images/Sentiment Scores by State, Trump.html')

fig_biden = px.choropleth(data_by_states,
                    locations = 'state_code',
                    locationmode = 'USA-states',
                    scope = 'usa',
                    color = 'Biden', labels = {'Biden': 'Sentiment score'},
                    range_color = (-1, 1),
                    hover_name = 'state_code',
                    hover_data = ['Biden'],
                    color_continuous_scale = 'blues',
                    title = 'US 2020 Elections: Biden Sentiment Scores by State'
)

plotly.offline.plot(fig_biden,
                    filename = 'images/Sentiment Scores by State, Biden.html')

# Checking the Predicted Winner and the Actual Winner by States
fig_predicted = px.choropleth(data_by_states,
                              locations='state_code',
                              locationmode='USA-states',
                              scope='usa',
                              color='won_predicted',  # Color by predicted winner
                              hover_name='state',
                              hover_data=['won_predicted'],
                              color_discrete_map={'Biden': 'blue', 'Trump': 'red'},
                              title='US 2020 Elections: Predicted Winner by State')

plotly.offline.plot(fig_predicted, filename = 'images/Predicted Winner by State.html')

fig_actual = px.choropleth(data_by_states,
                           locations='state_code',
                           locationmode='USA-states',
                           scope='usa',
                           color='won_actual',  # Color by actual winner
                           hover_name='state',
                           hover_data=['won_actual'],
                           color_discrete_map={'Biden': 'blue', 'Trump': 'red'},
                           title='US 2020 Elections: Actual Winner by State')

plotly.offline.plot(fig_actual, filename = 'images/Actual Winner by State.html')

# Predictions results
fig_results = px.choropleth(data_by_states,
                            locations='state_code',
                            locationmode='USA-states',
                            scope='usa',
                            color='prediction',
                            hover_name='state',
                            hover_data=['prediction'],
                            color_discrete_map={'Correct': 'green', 'Incorrect': 'red'},
                            title='US 2020 Elections: Correct Predictions by State')

plotly.offline.plot(fig_results, filename = 'images/Prediction Results by State.html')
