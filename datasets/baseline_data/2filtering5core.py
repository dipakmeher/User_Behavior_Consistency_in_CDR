import pandas as pd
import gzip
import json
import argparse

def extract_and_rename_columns(input_file):
    try:
        # Load the JSONL.GZ file
        records = []
        with gzip.open(input_file, 'rt', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                records.append(record)

        # Convert to DataFrame
        df = pd.DataFrame(records)

        # Select and rename the required columns
        df = df[['user_id', 'asin', 'rating']].rename(columns={
            'user_id': 'uid',
            'asin': 'iid',
            'rating': 'y'
        })

        print(f"Processed {len(df)} records from {input_file}")
        return df

    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return None

def filter_common_uids(book_df, movie_df, music_df, min_interactions=5):
    # Find common users (uids) across all three datasets
    common_uids = set(book_df['uid']).intersection(set(movie_df['uid']), set(music_df['uid']))

    # Filter each DataFrame to keep only users who are common to all datasets
    book_df = book_df[book_df['uid'].isin(common_uids)]
    movie_df = movie_df[movie_df['uid'].isin(common_uids)]
    music_df = music_df[music_df['uid'].isin(common_uids)]

    # Now apply the minimum interaction filter on the common users
    def filter_by_min_interactions(df):
        uid_counts = df.groupby('uid').size()
        valid_uids = uid_counts[uid_counts >= min_interactions].index
        return df[df['uid'].isin(valid_uids)]

    book_filtered = filter_by_min_interactions(book_df)
    movie_filtered = filter_by_min_interactions(movie_df)
    music_filtered = filter_by_min_interactions(music_df)

    # Re-calculate the overlapping users after applying the min interaction filter
    filtered_common_uids = set(book_filtered['uid']).intersection(set(movie_filtered['uid']), set(music_filtered['uid']))

    # Ensure the same set of users is present in all three filtered datasets
    book_filtered = book_filtered[book_filtered['uid'].isin(filtered_common_uids)]
    movie_filtered = movie_filtered[movie_filtered['uid'].isin(filtered_common_uids)]
    music_filtered = music_filtered[music_filtered['uid'].isin(filtered_common_uids)]

    return book_filtered, movie_filtered, music_filtered

def save_json_gz(dataframe, output_file):
    try:
        # Reorder columns to ensure 'uid', 'iid', 'y' order
        dataframe = dataframe[['uid', 'iid', 'y']]
        
        # Save the DataFrame as a new JSONL.GZ file
        with gzip.open(output_file, 'wt', encoding='utf-8') as f:
            for record in dataframe.to_dict(orient='records'):
                f.write(json.dumps(record) + '\n')

        print(f"Data saved to {output_file}")
    except Exception as e:
        print(f"Error saving {output_file}: {e}")

def generate_statistics(df, file_name, stats_file):
    try:
        length = len(df)
        unique_uids = df['uid'].nunique()
        unique_iids = df['iid'].nunique()

        # Write statistics to file
        with open(stats_file, 'a') as f:
            f.write(f"Statistics for {file_name}:\n")
            f.write(f"Total records (length): {length}\n")
            f.write(f"Unique uids: {unique_uids}\n")
            f.write(f"Unique iids: {unique_iids}\n")
            f.write(f"First 3 records:\n")
            f.write(f"{df.head(3).to_json(orient='records', lines=True)}\n")
            f.write("\n")

    except Exception as e:
        print(f"Error writing statistics for {file_name}: {e}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Filter datasets, rename columns, and generate statistics with consistent overlapping users.")

    # Define the input and output arguments
    parser.add_argument('--book-file', type=str, required=True, help="Path to the book input file (json.gz)")
    parser.add_argument('--movie-file', type=str, required=True, help="Path to the movie input file (json.gz)")
    parser.add_argument('--music-file', type=str, required=True, help="Path to the music input file (json.gz)")

    parser.add_argument('--book-output', type=str, required=True, help="Path to the output file for the filtered and renamed book data (json.gz)")
    parser.add_argument('--movie-output', type=str, required=True, help="Path to the output file for the filtered and renamed movie data (json.gz)")
    parser.add_argument('--music-output', type=str, required=True, help="Path to the output file for the filtered and renamed music data (json.gz)")

    parser.add_argument('--stats-file', type=str, required=True, help="Path to the statistics output file (txt)")

    parser.add_argument('--min-interactions', type=int, default=5, help="Minimum number of interactions per uid (default: 5)")

    # Parse the arguments
    args = parser.parse_args()

    # Clear the statistics file before writing
    with open(args.stats_file, 'w') as f:
        f.write("Statistics Report\n\n")

    # Load and rename the columns
    book_df = extract_and_rename_columns(args.book_file)
    movie_df = extract_and_rename_columns(args.movie_file)
    music_df = extract_and_rename_columns(args.music_file)

    # Ensure consistent overlapping users and apply filtering
    book_filtered, movie_filtered, music_filtered = filter_common_uids(book_df, movie_df, music_df, args.min_interactions)

    # Save the filtered datasets
    save_json_gz(book_filtered, args.book_output)
    save_json_gz(movie_filtered, args.movie_output)
    save_json_gz(music_filtered, args.music_output)

    # Generate statistics
    generate_statistics(book_filtered, 'Books', args.stats_file)
    generate_statistics(movie_filtered, 'Movies', args.stats_file)
    generate_statistics(music_filtered, 'Music', args.stats_file)

if __name__ == "__main__":
    main()
