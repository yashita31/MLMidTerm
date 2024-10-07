import subprocess
import re
import csv
from concurrent.futures import ThreadPoolExecutor
#import time

# List of models to use
models = [
    "llama3.2:1b",
    "gemma2:2b",
    "mistral-openorca",
    "neural-chat",
    "phi3"
]

# Function to read sentence starters from a file
def read_sentence_starters(input_file):
    """Read sentence starters from a file."""
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as file:
        sentence_starters = [line.strip() for line in file.readlines() if line.strip()]
    return sentence_starters

# Function to clean output by filtering out unwanted lines
def clean_output(output):
    """Remove 'failed to get console mode' lines and return clean output."""
    lines = output.splitlines()
    filtered_lines = [line for line in lines if not line.startswith("failed to get console mode")]
    return "\n".join(filtered_lines)

# Function to generate completions using different LLMs
def generate_sentence_completion(sentence_starts, model, word_limit=20, output_file="output_6llms.csv"):
    for sentence in sentence_starts:
        # Explicit prompt for text completion
        prompt = f'This is a text completion task. Make any assumptions. Complete the second part of the sentence. The output should be only the second part of the text completion and nothing else. The first part of the sentence is: "{sentence}"'

        # Command to run the model using ollama
        command = ["ollama", "run", model, prompt]

        try:
            #start_time = time.time()
            # Use subprocess to call the specific model and generate the completion
            result = subprocess.run(command, capture_output=True, text=True, shell=True)

            if result.returncode == 0:
                # Clean and process the result
                completion = clean_output(result.stdout.strip())

                # Limit the completion by word count
                words = completion.split()
                limited_completion = " ".join(words[:word_limit])

                # Truncate at first sentence-ending punctuation
                # sentence_ending_match = re.search(r'([.!?])', limited_completion)
                # if sentence_ending_match:
                #     limited_completion = limited_completion[:sentence_ending_match.end()]

                # Write to output CSV
                with open(output_file, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([sentence, limited_completion, model])
            else:
                print(f"Error generating completion for {sentence} using {model}: {result.stderr}")

            #end_time = time.time()
            #time_elapsed = end_time-start_time
            #print(f"Time while runnning {model}: {time_elapsed}")
        except Exception as e:
            print(f"Exception while running model {model}: {str(e)}")
    
    print(f"Completions for {model} saved to {output_file}")

# Main function to generate completions for all models with thread management
def main():
    input_file = "data_small.txt"  # Path to your input file containing sentence starters
    word_limit = 30  # Adjust word limit if needed
    output_file = "output.csv"  # Combined output file for all models
    max_threads = 8  # Set the maximum number of threads (or models) to run concurrently

    # Read the sentence starters from the input file
    sentence_starts = read_sentence_starters(input_file)

    # Write header to the CSV file before starting threads
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["xi", "xj", "model"])  # CSV header

    # Use ThreadPoolExecutor to limit the number of concurrent threads
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(generate_sentence_completion, sentence_starts, model, word_limit, output_file) for model in models]

        # Wait for all futures to complete
        for future in futures:
            future.result()

# Call the main function
if __name__ == "__main__":
    main()
