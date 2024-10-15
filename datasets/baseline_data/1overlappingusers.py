import pandas as pd
import gzip
import json
import argparse

def load_jsonl_gz(file_path):
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        print(f"Loaded {len(data)} records from {file_path}")
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

def save_json_gz(dataframe, file_path):
    try:
        with gzip.open(file_path, 'wt', encoding='utf-8') as f:
            for record in dataframe.to_dict(orient='records'):
                f.write(json.dumps(record) + '\n')
        print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error saving {file_path}: {e}")

def calculate_stats(dataframe):
    length = len(dataframe)
    unique_user_ids = dataframe['user_id'].nunique()
    unique_asins = dataframe['asin'].nunique()
    return length, unique_user_ids, unique_asins

def filter_overlapping_users(book_file, movie_file, music_file, book_output, movie_output, music_output, summary_file):
    # Load all three files
    book_data = load_jsonl_gz(book_file)
    movie_data = load_jsonl_gz(movie_file)
    music_data = load_jsonl_gz(music_file)

    if book_data.empty or movie_data.empty or music_data.empty:
        print("One of the datasets is empty. Exiting.")
        return

    # Find overlapping user_ids in all three datasets
    overlapping_user_ids = set(book_data['user_id']).intersection(set(movie_data['user_id'])).intersection(set(music_data['user_id']))
    print(f"Found {len(overlapping_user_ids)} overlapping user_ids across all three datasets")

    # Filter out records with overlapping user_ids
    book_overlap = book_data[book_data['user_id'].isin(overlapping_user_ids)]
    movie_overlap = movie_data[movie_data['user_id'].isin(overlapping_user_ids)]
    music_overlap = music_data[music_data['user_id'].isin(overlapping_user_ids)]

    # Save the filtered data to json.gz
    save_json_gz(book_overlap, book_output)
    save_json_gz(movie_overlap, movie_output)
    save_json_gz(music_overlap, music_output)

    # Calculate statistics for the resultant data
    book_length, book_unique_user_ids, book_unique_asins = calculate_stats(book_overlap)
    movie_length, movie_unique_user_ids, movie_unique_asins = calculate_stats(movie_overlap)
    music_length, music_unique_user_ids, music_unique_asins = calculate_stats(music_overlap)

    # Write the results to a summary file
    with open(summary_file, 'w') as f:
        f.write(f"Summary of results after filtering:\n")
        f.write(f"\nBook Data ({book_output}):\n")
        f.write(f"Total records: {book_length}\n")
        f.write(f"Unique user IDs: {book_unique_user_ids}\n")
        f.write(f"Unique ASINs: {book_unique_asins}\n")

        f.write(f"\nMovie Data ({movie_output}):\n")
        f.write(f"Total records: {movie_length}\n")
        f.write(f"Unique user IDs: {movie_unique_user_ids}\n")
        f.write(f"Unique ASINs: {movie_unique_asins}\n")

        f.write(f"\nMusic Data ({music_output}):\n")
        f.write(f"Total records: {music_length}\n")
        f.write(f"Unique user IDs: {music_unique_user_ids}\n")
        f.write(f"Unique ASINs: {music_unique_asins}\n")

    print(f"Summary of results has been written to {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Filter overlapping users between three JSONL.GZ files and save the results as JSON.GZ. Additionally, output the summary of results in a text file.")
    parser.add_argument('--book-file', type=str, required=True, help="Path to the Book data file (jsonl.gz).")
    parser.add_argument('--movie-file', type=str, required=True, help="Path to the Movie data file (jsonl.gz).")
    parser.add_argument('--music-file', type=str, required=True, help="Path to the Music data file (jsonl.gz).")
    parser.add_argument('--book-output', type=str, required=True, help="Path to save the filtered Book data (json.gz).")
    parser.add_argument('--movie-output', type=str, required=True, help="Path to save the filtered Movie data (json.gz).")
    parser.add_argument('--music-output', type=str, required=True, help="Path to save the filtered Music data (json.gz).")
    parser.add_argument('--summary-file', type=str, required=True, help="Path to save the summary of results (txt).")

    args = parser.parse_args()

    filter_overlapping_users(args.book_file, args.movie_file, args.music_file, args.book_output, args.movie_output, args.music_output, args.summary_file)

if __name__ == "__main__":
    main()
