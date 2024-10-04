import pandas as pd

def cleaning(df, candidate):
    df = df[(df['country'] == 'United States of America') | (df['country'] == 'United States')] # only USA
    df = df.dropna() # removing all the na's
    df = df[['created_at', 'tweet', 'state_code']] # selecting only relevant columns
    df['candidate'] = candidate # creating candidate column
    df['created_at'] = pd.to_datetime(df['created_at'], errors = 'coerce') # formatting as pd datetime
    df = df[(df['tweet'].str.len() >= 10)] # remove meaningless tweets (tweets that are less than 10 characters)
    # didn't remove replies, maybe I should, gotta think about it
    return df

