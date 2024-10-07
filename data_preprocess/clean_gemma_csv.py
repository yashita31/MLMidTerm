import pandas as pd
import re

# Function to remove special characters
def remove_special_characters(text):
    if isinstance(text, str):
        # Replace all non-alphanumeric characters (except spaces) with an empty string
        return re.sub(r'[^A-Za-z0-9\s]+', '', text)
    else:
        return text  # Return the value as is if it's not a string

# Load CSV file
input_csv = 'data/gpt-neo-4k.csv'  # Replace with your actual file path
df = pd.read_csv(input_csv, encoding='utf-8')

# Assuming the second column is index 1 (0-based indexing), remove special characters from this column
df.iloc[:, 1] = df.iloc[:, 1].apply(remove_special_characters)

# Save to a new CSV file
output_csv = 'data/gpt_neo_modified.csv'  # Replace with desired output file path
df.to_csv(output_csv, index=False)

print(f"Special characters removed and data saved to {output_csv}")
