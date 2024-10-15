import pandas as pd
import argparse
import gzip
import json

def filter_and_rearrange(file_path, output_file_path, user_id, reordered_columns):
    # Initialize an empty list to store the filtered records
    filtered_records = []

    # Open the gzipped JSON file
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        # Read the file line by line
        for line in f:
            # Parse each line as a JSON object
            record = json.loads(line)

            # Check if the record belongs to the specified user_id
            if record['user_id'] == user_id:
                filtered_records.append(record)

    # Convert the filtered records to a DataFrame
    user_records = pd.DataFrame(filtered_records)

    # Rearrange the columns
    user_records = user_records[reordered_columns]

    # Save the rearranged DataFrame to a new CSV file
    user_records.to_csv(output_file_path, index=False)

    print(f'Filtered and rearranged data saved to {output_file_path}')

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Filter and rearrange records for a specific user.')

    parser.add_argument('--input', type=str, required=True, help='Path to the input gzipped JSONL file')
    parser.add_argument('--output', type=str, required=True, help='Path to the output CSV file')
    parser.add_argument('--user_id', type=str, required=True, help='User ID to filter records by')

    args = parser.parse_args()

    # Define the desired column order (reordered_columns)
    reordered_columns = ['user_id', 'asin', 'rating', 'title', 'text', 'images', 'parent_asin', 'features', 'description', 'price']

    # Call the function with the provided arguments
    filter_and_rearrange(args.input, args.output, args.user_id, reordered_columns)
