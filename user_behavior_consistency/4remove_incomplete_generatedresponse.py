import pandas as pd
import argparse

def filter_responses(input_file, output_file):
    # Load the CSV file
    df = pd.read_csv(input_file)

    # Filter out records where either column has the incomplete response
    filtered_df = df[
        (df['movie_generated_response'] != 'Failed to generate a valid response') |
        (df['book_generated_response'] != 'Failed to generate a valid response')
    ]

    # Save the remaining records to a new CSV file
    filtered_df.to_csv(output_file, index=False)

    print(f'Remaining records have been saved to {output_file}')

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Filter out records with incomplete responses.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the filtered output CSV file')

    args = parser.parse_args()

    # Call the filter_responses function with the provided arguments
    filter_responses(args.input_file, args.output_file)

