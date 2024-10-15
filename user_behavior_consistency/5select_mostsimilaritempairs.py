import pandas as pd
import argparse

def select_highest_similarity(input_file, output_file):
    # Load the filtered CSV file
    df = pd.read_csv(input_file)

    # Group by 'book_asin' and select the first record of each group with the highest similarity
    highest_similarity_df = df.sort_values('similarity', ascending=False).drop_duplicates(subset=['book_asin'], keep='first')

    # Save the highest similarity records to a new CSV file
    highest_similarity_df.to_csv(output_file, index=False)

    print(f'Highest similarity records have been saved to {output_file}')

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Select highest similarity records for each book ASIN.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input filtered CSV file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output CSV file with highest similarity records')

    args = parser.parse_args()

    # Call the select_highest_similarity function with the provided arguments
    select_highest_similarity(args.input_file, args.output_file)

