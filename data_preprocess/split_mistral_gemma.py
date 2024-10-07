import pandas as pd

# Load the CSV file
df = pd.read_csv('data/output_split1.csv')  # Replace 'your_file.csv' with your actual file path

# Filter the rows for each model
mistral_df = df[df['model'] == 'mistral-openorca'].iloc[:4000]
gemma2_df = df[df['model'] == 'gemma2:2b'].iloc[:4000]

# Remove the 'model' column from both dataframes
mistral_df = mistral_df.drop(columns=['model'])
gemma2_df = gemma2_df.drop(columns=['model'])

# Save the first 4k rows of each model to separate CSV files without any labels
mistral_df.to_csv('data/mistral_openorca_4k.csv', index=False, header=False)
gemma2_df.to_csv('data/gemma2_2b_4k.csv', index=False, header=False)

print("Files saved successfully!")
