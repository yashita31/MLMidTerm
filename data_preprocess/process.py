import pandas as pd
import re

def clean_text(text):
    # Remove "Replying to" and the following text (including "and X others")
    text = re.sub(r'Replying to( and \d+ others)?\s*', '', text)
    
    # Remove standalone "others"
    text = re.sub(r'\bothers\b', '', text)  # Removes the word "others"
    
    # Remove standalone numbers (including those with "K")
    text = re.sub(r'\b\d+[Kk]?\b', '', text)  # Removes numbers like "1", "2K", "500", etc.
    
    # Remove "and" that is at the start of a sentence or followed by whitespace
    text = re.sub(r'\band\s+', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove @mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove emojis
    emoji_pattern = re.compile(
        "[" 
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF"  
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F700-\U0001F77F"  
        u"\U0001F780-\U0001F7FF"  
        u"\U0001F800-\U0001F8FF"  
        u"\U0001F900-\U0001F9FF"  
        u"\U0001FA00-\U0001FA6F"  
        u"\U0001FA70-\U0001FAFF"  
        u"\U00002702-\U000027B0"  
        u"\U000024C2-\U0001F251"  
        "]+", flags=re.UNICODE)
    
    text = emoji_pattern.sub(r'', text)  
    
    # Remove special characters (excluding spaces)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)

    text = re.sub(r'RT ', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# Load the CSV file
file_path = 'data/elonmusk.csv'
tweets_df = pd.read_csv(file_path)

# Apply the cleaning function to the 'text' column
tweets_df['cleaned_text'] = tweets_df['text'].apply(clean_text)

# Filter tweets with more than 5 words after cleaning
tweets_df['word_count'] = tweets_df['cleaned_text'].apply(lambda x: len(x.split()))
filtered_tweets = tweets_df[tweets_df['word_count'] > 5]

# Display the filtered and cleaned tweets
print(filtered_tweets[['cleaned_text']])

# Optionally, save the cleaned and filtered tweets to a new CSV
filtered_tweets[['cleaned_text']].to_csv('data/cleaned_filtered_tweets.csv', index=False)
