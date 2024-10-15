import pandas as pd
import argparse

# Function to remove leading zeros while preserving trailing zeros
def remove_leading_zeros(asin):
    return asin.lstrip('0')

def merge_data(similarity_file, books_sentiment_file, movies_sentiment_file, output_file):
    # Load the main similarity data
    similarity_df = pd.read_csv(similarity_file)

    # Load the sentiment data for books and movies
    books_sentiment_df = pd.read_csv(books_sentiment_file)
    movies_sentiment_df = pd.read_csv(movies_sentiment_file)

    # Ensure ASINs are treated as strings and remove leading/trailing spaces
    similarity_df['book_asin'] = similarity_df['book_asin'].astype(str).str.strip().str.upper().apply(remove_leading_zeros)
    similarity_df['movie_asin'] = similarity_df['movie_asin'].astype(str).str.strip().str.upper().apply(remove_leading_zeros)

    books_sentiment_df['asin'] = books_sentiment_df['asin'].astype(str).str.strip().str.upper().apply(remove_leading_zeros)
    movies_sentiment_df['asin'] = movies_sentiment_df['asin'].astype(str).str.strip().str.upper().apply(remove_leading_zeros)

    # Merge the sentiment-generated responses into the similarity data based on book ASIN
    merged_df = pd.merge(similarity_df, books_sentiment_df[['asin', 'Generated_Response']], left_on='book_asin', right_on='asin', how='left')
    merged_df = merged_df.rename(columns={'Generated_Response': 'book_sentiment_response'})
    merged_df = merged_df.drop(columns=['asin'])

    # Merge the sentiment-generated responses into the similarity data based on movie ASIN
    merged_df = pd.merge(merged_df, movies_sentiment_df[['asin', 'Generated_Response']], left_on='movie_asin', right_on='asin', how='left')
    merged_df = merged_df.rename(columns={'Generated_Response': 'movie_sentiment_response'})
    merged_df = merged_df.drop(columns=['asin'])

    # Save the final dataframe to a new CSV file
    merged_df.to_csv(output_file, index=False)

    print(f"Merging complete. The result has been saved to '{output_file}'.")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Merge similarity data with sentiment responses.')
    parser.add_argument('--similarity_file', type=str, required=True, help='Path to the similarity CSV file')
    parser.add_argument('--books_sentiment_file', type=str, required=True, help='Path to the books sentiment CSV file')
    parser.add_argument('--movies_sentiment_file', type=str, required=True, help='Path to the movies sentiment CSV file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the merged CSV file')

    args = parser.parse_args()

    # Call the merge_data function with the provided arguments
    merge_data(args.similarity_file, args.books_sentiment_file, args.movies_sentiment_file, args.output_file)

