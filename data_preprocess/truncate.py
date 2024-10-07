import pandas as pd

# Load the CSV without headers
file_path = 'data/cleaned_donald_filtered_tweets.csv'  # Adjust this if needed
df = pd.read_csv(file_path, header=None, names=['tweet'])

# Function to truncate text to the first 3 words
def truncate_to_three_words(text):
    words = text.split()  # Split the text into words
    return ' '.join(words[:3])  # Join the first 3 words

# Apply the truncation function to each tweet
df['truncated_text'] = df['tweet'].apply(truncate_to_three_words)

# Save the result to a new CSV file
output_file = 'data/truncated_donald_tweets.csv'
df[['truncated_text']].to_csv(output_file, index=False, header=False)

print(f"Truncated texts saved to {output_file}")
