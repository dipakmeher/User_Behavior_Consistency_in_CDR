import pandas as pd
import gzip
import json
import argparse
import os

def load_jsonl_gz(file_path):
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]  # Load line by line for jsonl.gz
        print(f"Loaded {len(data)} records from {file_path}")
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

def save_json_gz(dataframe, file_path):
    try:
        # Ensure the directory exists
        output_dir = os.path.dirname(file_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with gzip.open(file_path, 'wt', encoding='utf-8') as f:
            for record in dataframe.to_dict(orient='records'):
                f.write(json.dumps(record) + '\n')  # Write each record as a line (jsonl)
        print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error saving {file_path}: {e}")

def save_csv(dataframe, file_path):
    try:
        # Ensure the directory exists
        output_dir = os.path.dirname(file_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the DataFrame as a CSV file
        dataframe.to_csv(file_path, index=False)  # Save CSV without index
        print(f"Data saved to {file_path}")

    except Exception as e:
        print(f"Error saving {file_path}: {e}")

def calculate_stats(dataframe):
    length = len(dataframe)
    unique_uids = dataframe['uid'].nunique()  # Changed from 'user_id' to 'uid'
    unique_iids = dataframe['iid'].nunique()  # Changed from 'asin' to 'iid'
    return length, unique_uids, unique_iids

def create_test_dataset(book_file, movie_file, music_file, book_test_output, movie_test_output, music_test_output, stats_output, max_records=1000):
    # Load all overlapping files
    book_data = load_jsonl_gz(book_file)
    movie_data = load_jsonl_gz(movie_file)
    music_data = load_jsonl_gz(music_file)

    if book_data.empty or movie_data.empty or music_data.empty:
        print("One of the datasets is empty. Exiting.")
        return

    # Find unique overlapping uids (changed from 'user_id')
    overlapping_uids = set(book_data['uid']).intersection(set(movie_data['uid'])).intersection(set(music_data['uid']))

    # DataFrames to store the test data
    book_test = pd.DataFrame()
    movie_test = pd.DataFrame()
    music_test = pd.DataFrame()

    # Counter to keep track of total records added
    total_records = 0

    # Process each overlapping user
    for uid in overlapping_uids:  # Changed 'user_id' to 'uid'
        # Get user records in each dataset
        user_book_records = book_data[book_data['uid'] == uid]  # Changed 'user_id' to 'uid'
        user_movie_records = movie_data[movie_data['uid'] == uid]  # Changed 'user_id' to 'uid'
        user_music_records = music_data[music_data['uid'] == uid]  # Changed 'user_id' to 'uid'

        # Find the minimum number of records among the three domains
        min_records = min(len(user_book_records), len(user_movie_records), len(user_music_records))

        # Select the minimum number of records from each domain, but ensure we don't exceed the max_records limit
        records_to_add = min(min_records, max_records - total_records)

        if records_to_add <= 0:
            break  # Stop if we have reached the required 1000 records

        # Select records and add them to the test sets
        book_test = pd.concat([book_test, user_book_records.head(records_to_add)])
        movie_test = pd.concat([movie_test, user_movie_records.head(records_to_add)])
        music_test = pd.concat([music_test, user_music_records.head(records_to_add)])

        total_records += records_to_add

        if total_records >= max_records:
            break  # Stop once we've collected enough records

    # Ensure the correct column order: uid, iid, y
    book_test = book_test[['uid', 'iid', 'y']]
    movie_test = movie_test[['uid', 'iid', 'y']]
    music_test = music_test[['uid', 'iid', 'y']]

    print(f"Collected {total_records} records for the test dataset.")

    # Save the test datasets to JSON.GZ files
    #save_json_gz(book_test, book_test_output)
    #save_json_gz(movie_test, movie_test_output)
    #save_json_gz(music_test, music_test_output)

    save_csv(book_test, book_test_output)
    save_csv(movie_test, movie_test_output)
    save_csv(music_test, music_test_output)
    print(f"Test datasets created and saved for Books, Movies, and Music.")

    # Calculate statistics for each dataset and write to a text file
    book_length, book_unique_uids, book_unique_iids = calculate_stats(book_test)  # Changed to uid and iid
    movie_length, movie_unique_uids, movie_unique_iids = calculate_stats(movie_test)  # Changed to uid and iid
    music_length, music_unique_uids, music_unique_iids = calculate_stats(music_test)  # Changed to uid and iid

    with open(stats_output, 'w') as f:
        f.write(f"Statistics for Test Datasets\n")
        f.write(f"\nBook Data ({book_test_output}):\n")
        f.write(f"Total records: {book_length}\n")
        f.write(f"Unique uids: {book_unique_uids}\n")  # Changed to uid
        f.write(f"Unique iids: {book_unique_iids}\n")  # Changed to iid

        f.write(f"\nMovie Data ({movie_test_output}):\n")
        f.write(f"Total records: {movie_length}\n")
        f.write(f"Unique uids: {movie_unique_uids}\n")  # Changed to uid
        f.write(f"Unique iids: {movie_unique_iids}\n")  # Changed to iid

        f.write(f"\nMusic Data ({music_test_output}):\n")
        f.write(f"Total records: {music_length}\n")
        f.write(f"Unique uids: {music_unique_uids}\n")  # Changed to uid
        f.write(f"Unique iids: {music_unique_iids}\n")  # Changed to iid

    print(f"Statistics written to {stats_output}")

def main():
    parser = argparse.ArgumentParser(description="Create test datasets from overlapping user records across Books, Movies, and Music, and generate statistics.")
    parser.add_argument('--book-file', type=str, required=True, help="Path to the overlapping Books data file (jsonl.gz).")
    parser.add_argument('--movie-file', type=str, required=True, help="Path to the overlapping Movies data file (jsonl.gz).")
    parser.add_argument('--music-file', type=str, required=True, help="Path to the overlapping Music data file (jsonl.gz).")
    parser.add_argument('--book-test-output', type=str, required=True, help="Path to save the test Books data (json.gz).")
    parser.add_argument('--movie-test-output', type=str, required=True, help="Path to save the test Movies data (json.gz).")
    parser.add_argument('--music-test-output', type=str, required=True, help="Path to save the test Music data (json.gz).")
    parser.add_argument('--stats-output', type=str, required=True, help="Path to save the statistics file (txt).")
    parser.add_argument('--max-records', type=int, default=1000, help="Maximum number of records to collect for the test dataset.")

    args = parser.parse_args()

    create_test_dataset(args.book_file, args.movie_file, args.music_file, args.book_test_output, args.movie_test_output, args.music_test_output, args.stats_output, args.max_records)

if __name__ == "__main__":
    main()
