import pandas as pd
import argparse
import re

# Function to map correlation labels to numbers
def map_correlation(correlation_text):
    # Define the mapping
    if "Highly-consistent" in correlation_text:
        return 1
    elif "Neutral" in correlation_text:
        return 0
    elif "Not-consistent" in correlation_text:
        return -1
    else:
        return None  # In case no valid label is found

def process_correlation(file_path, output_file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Apply the mapping function to the 'user_correlation' column
    df['correlation_score'] = df['user_correlation'].apply(map_correlation)

    # Reorder columns to place 'correlation_score' next to 'user_correlation'
    columns = df.columns.tolist()
    # Find the position of 'user_correlation'
    user_correlation_index = columns.index('user_correlation')
    # Insert 'correlation_score' after 'user_correlation'
    columns.insert(user_correlation_index + 1, columns.pop(columns.index('correlation_score')))
    df = df[columns]

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_file_path, index=False)

    print(f"New file with correlation scores saved as {output_file_path}")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process correlation results and map them to scores.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output CSV file')

    args = parser.parse_args()

    # Call the process_correlation function with the provided arguments
    process_correlation(args.input_file, args.output_file)

