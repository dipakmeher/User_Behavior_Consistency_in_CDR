import argparse
import gzip
import json
from collections import defaultdict

# Function to filter data and log file details
def filter_data(input_jsonl_gz_file, filtered_jsonl_gz_file, min_interactions, log_file):
    # Count user interactions and unique user_id, asin values
    user_interactions = defaultdict(int)
    unique_users = set()
    unique_asins = set()

    # First pass: count interactions and track unique user_id and asin
    with gzip.open(input_jsonl_gz_file, 'rt', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            row = json.loads(line.strip())
            user_id = row.get('user_id')
            asin = row.get('asin')
            if user_id:
                user_interactions[user_id] += 1
                unique_users.add(user_id)
            if asin:
                unique_asins.add(asin)

    original_data_length = len(user_interactions)

    # Filter users with at least min_interactions
    filtered_data = []
    filtered_users = set()
    filtered_asins = set()

    with gzip.open(input_jsonl_gz_file, 'rt', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            row = json.loads(line.strip())
            user_id = row.get('user_id')
            asin = row.get('asin')
            if user_id and user_interactions[user_id] >= min_interactions:
                filtered_data.append(row)
                filtered_users.add(user_id)
                if asin:
                    filtered_asins.add(asin)

    filtered_data_length = len(filtered_data)

    # Write filtered data to a new jsonl.gz file
    with gzip.open(filtered_jsonl_gz_file, 'wt', encoding='utf-8') as gzfile:
        for row in filtered_data:
            gzfile.write(json.dumps(row) + '\n')

    # Log original and filtered file information
    with open(log_file, 'w') as log:
        log.write(f"Original file length: {original_data_length} records\n")
        log.write(f"Unique user_ids in original file: {len(unique_users)}\n")
        log.write(f"Unique asins in original file: {len(unique_asins)}\n")
        log.write(f"Filtered file length: {filtered_data_length} records\n")
        log.write(f"Unique user_ids in filtered file: {len(filtered_users)}\n")
        log.write(f"Unique asins in filtered file: {len(filtered_asins)}\n")

    print(f"Filtered data has been written to {filtered_jsonl_gz_file}")
    print(f"Log file created at {log_file}")

# Main function to parse arguments
def main():
    parser = argparse.ArgumentParser(description="Filter users by the number of interactions in a JSONL.GZ file and log the results.")
    parser.add_argument('--input_jsonl_gz', type=str, required=True, help="Path to the input jsonl.gz file.")
    parser.add_argument('--output_jsonl_gz', type=str, required=True, help="Path to the output filtered jsonl.gz file.")
    parser.add_argument('--min_interactions', type=int, required=True, help="Minimum number of interactions for filtering users.")
    parser.add_argument('--log_file', type=str, required=True, help="Path to save the log file.")

    args = parser.parse_args()

    # Call the filter function with the provided arguments
    filter_data(args.input_jsonl_gz, args.output_jsonl_gz, args.min_interactions, args.log_file)

if __name__ == "__main__":
    main()
