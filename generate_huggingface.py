from transformers import pipeline
import pandas as pd
import torch
from datasets import Dataset

# Check which device is being used
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load your dataset
data = pd.read_csv('data/Dataset10k.csv')

# Convert the DataFrame to a Hugging Face Dataset
hf_dataset = Dataset.from_pandas(data)

# Define a list of five models to use
model_names = [
    'gpt2',
    'EleutherAI/gpt-neo-125M',
    'facebook/opt-125m',
]


# Set up text generation pipelines for each model with the specified device
generators = [
    pipeline('text-generation', model=model, device=0 if device == 'cuda' else -1, pad_token_id=50256)
    for model in model_names
]

# Generate completions for each model and save the results in separate CSV files
for model_name, generator in zip(model_names, generators):
    # Function to generate completion for a batch of prompts
    def generate_batch_completions(batch):
        prompts = batch['Please ignore prior']  # Already a list
        completions = generator(prompts, max_length=25, num_return_sequences=1)  # Removed truncation
        return {'Completion': [result[0]['generated_text'] for result in completions]}

    # Apply the function to the dataset
    completed_data = hf_dataset.map(generate_batch_completions, batched=True)

    # Convert back to DataFrame
    completed_df = completed_data.to_pandas()

    # Save the completed dataset to a separate file for each model
    safe_model_name = model_name.replace('/', '_')
    output_filename = f'data/completed_dataset10k_{safe_model_name}.csv'
    completed_df.to_csv(output_filename, index=False)

    print(f"Completions for model '{model_name}' saved to '{output_filename}'")





