import pandas as pd
import argparse

def process_data(similarity_file, books_file, movies_file, output_books_file, output_movies_file):
    # Read the filtered ASINs from the first file
    similarity_df = pd.read_csv(similarity_file)
    print("Similarity DataFrame shape:", similarity_df.shape)
    print("Similarity DataFrame head:\n", similarity_df.head())

    # Ensure no leading/trailing spaces and consistent casing, converting to string to avoid errors
    similarity_df['book_asin'] = similarity_df['book_asin'].astype(str).str.strip().str.upper()
    similarity_df['movie_asin'] = similarity_df['movie_asin'].astype(str).str.strip().str.upper()

    # Remove leading zeros from ASINs in similarity_df to match the format of oneuser_books_df and oneuser_movies_df
    similarity_df['book_asin'] = similarity_df['book_asin'].str.lstrip('0')
    similarity_df['movie_asin'] = similarity_df['movie_asin'].str.lstrip('0')

    # Print unique ASINs to verify they are correct after removing leading zeros
    print("Unique Book ASINs in Similarity DataFrame after removing leading zeros:")
    print(similarity_df['book_asin'].unique())

    print("\nUnique Movie ASINs in Similarity DataFrame after removing leading zeros:")
    print(similarity_df['movie_asin'].unique())

    # Extract the unique book and movie ASINs
    filtered_book_asins = similarity_df['book_asin'].unique()
    filtered_movie_asins = similarity_df['movie_asin'].unique()
    print("Filtered Book ASINs (sample):\n", filtered_book_asins[:10])
    print("Filtered Movie ASINs (sample):\n", filtered_movie_asins[:10])

    # Read the full book and movie ASINs from the other two files
    oneuser_books_df = pd.read_csv(books_file, dtype={'asin': str})
    oneuser_movies_df = pd.read_csv(movies_file, dtype={'asin': str})

    # Ensure no leading/trailing spaces and consistent casing, converting to string to avoid errors
    oneuser_books_df['asin'] = oneuser_books_df['asin'].astype(str).str.strip().str.upper()
    oneuser_movies_df['asin'] = oneuser_movies_df['asin'].astype(str).str.strip().str.upper()

    # Remove leading zeros from ASINs in oneuser_books_df and oneuser_movies_df
    oneuser_books_df['asin'] = oneuser_books_df['asin'].str.lstrip('0')
    oneuser_movies_df['asin'] = oneuser_movies_df['asin'].str.lstrip('0')

    # Check the shapes of the original dataframes
    print("Oneuser Books DataFrame shape:", oneuser_books_df.shape)
    print("Oneuser Movies DataFrame shape:", oneuser_movies_df.shape)

    print("Oneuser Books DataFrame head:\n", oneuser_books_df.head())
    print("Oneuser Movies DataFrame head:\n", oneuser_movies_df.head())

    # Check for overlap between filtered ASINs and the full dataset ASINs
    book_overlap = set(filtered_book_asins).intersection(set(oneuser_books_df['asin']))
    movie_overlap = set(filtered_movie_asins).intersection(set(oneuser_movies_df['asin']))
    print("Number of overlapping book ASINs:", len(book_overlap))
    print("Number of overlapping movie ASINs:", len(movie_overlap))

    # Debug: Print some overlapping ASINs
    print("Overlapping book ASINs (sample):\n", list(book_overlap)[:10])
    print("Overlapping movie ASINs (sample):\n", list(movie_overlap)[:10])

    # Filter the dataframes based on the ASINs from the first file
    filtered_books_df = oneuser_books_df[oneuser_books_df['asin'].isin(filtered_book_asins)]
    filtered_movies_df = oneuser_movies_df[oneuser_movies_df['asin'].isin(filtered_movie_asins)]

    # Print the shape of the filtered DataFrames to ensure columns are not lost
    print("Filtered Books DataFrame shape:", filtered_books_df.shape)
    print("Filtered Movies DataFrame shape:", filtered_movies_df.shape)
    # Check the head of the filtered DataFrames
    print("Filtered Books DataFrame head:\n", filtered_books_df.head())
    print("Filtered Movies DataFrame head:\n", filtered_movies_df.head())

    # Save the filtered dataframes to new CSV files
    filtered_books_df.to_csv(output_books_file, index=False)
    filtered_movies_df.to_csv(output_movies_file, index=False)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process and filter ASIN data from CSV files.')
    parser.add_argument('--similarity_file', type=str, required=True, help='Path to the similarity CSV file')
    parser.add_argument('--books_file', type=str, required=True, help='Path to the books CSV file')
    parser.add_argument('--movies_file', type=str, required=True, help='Path to the movies CSV file')
    parser.add_argument('--output_books_file', type=str, required=True, help='Path to save the filtered books CSV file')
    parser.add_argument('--output_movies_file', type=str, required=True, help='Path to save the filtered movies CSV file')

    args = parser.parse_args()

    # Call the process_data function with the provided arguments
    process_data(args.similarity_file, args.books_file, args.movies_file, args.output_books_file, args.output_movies_file)

