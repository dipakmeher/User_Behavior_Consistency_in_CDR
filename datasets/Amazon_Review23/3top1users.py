import pandas as pd
import argparse
import gzip
import json

# Function to process the data and generate log information
def process_top_user(input_jsonl_gz_file, output_csv_file, log_file):
    # Load the JSONL.GZ file into a pandas DataFrame
    data = []
    with gzip.open(input_jsonl_gz_file, 'rt', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            data.append(json.loads(line.strip()))
    
    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(data)

    # Ensure the column names are as expected
    print(f"Columns in the input data: {df.columns.tolist()}")

    # Count interactions for each user
    user_interactions = df['user_id'].value_counts()

    # Get the top user with maximum interactions
    top_user = user_interactions.head(1).index.tolist()

    # Select all records that belong to this top user
    top_user_records = df[df['user_id'].isin(top_user)]

    # Reorder the columns (ensure the columns exist in the DataFrame)
    reordered_columns = ['user_id', 'asin', 'rating', 'title', 'text', 'images', 'parent_asin', 'features', 'description', 'price']
    top_user_records = top_user_records[reordered_columns]

    # Save these records to a new CSV file
    top_user_records.to_csv(output_csv_file, index=False)

    # Log original and filtered file information
    with open(log_file, 'w') as log:
        log.write(f"Length of original input file: {len(df)} records\n")
        log.write(f"Unique user_ids in original file: {df['user_id'].nunique()}\n")
        log.write(f"Unique asins in original file: {df['asin'].nunique()}\n")
        
        log.write(f"Length of output file (top user): {len(top_user_records)} records\n")
        log.write(f"Unique user_ids in output file: {top_user_records['user_id'].nunique()}\n")
        log.write(f"Unique asins in output file: {top_user_records['asin'].nunique()}\n")

    # Display the top user and the number of records selected
    print(f"Top user: {top_user}")
    for user in top_user:
        user_record_count = len(top_user_records[top_user_records['user_id'] == user])
        print(f"User {user} has {user_record_count} records.")

    # Display the total number of records selected
    print(f"Total number of records selected: {len(top_user_records)}")

# Main function to parse arguments
def main():
    parser = argparse.ArgumentParser(description="Process top user data from a JSONL.GZ file and output to CSV with logging.")
    parser.add_argument('--input_jsonl_gz', type=str, required=True, help="Path to the input JSONL.GZ file.")
    parser.add_argument('--output_csv', type=str, required=True, help="Path to the output CSV file.")
    parser.add_argument('--log_file', type=str, required=True, help="Path to the log file.")

    args = parser.parse_args()

    # Call the process function with the provided arguments
    process_top_user(args.input_jsonl_gz, args.output_csv, args.log_file)

if __name__ == "__main__":
    main()
