import pandas as pd
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity

# Function to parse embedding strings to float arrays
def parse_embedding(embedding_str):
    try:
        embedding_str = embedding_str.strip('[]')
        embedding_list = list(map(float, embedding_str.split()))
        return np.array(embedding_list)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Error parsing embedding: {embedding_str}\n{e}")

def calculate_similarity(embedding1, embedding2):
    # Using sklearn's cosine_similarity to calculate similarity
    if embedding1.size == 0 or embedding2.size == 0:
        raise ValueError("One of the embeddings is empty.")
    return cosine_similarity([embedding1], [embedding2])[0][0]

def main(books_file, movies_file, output_file_1, output_file_2, all_results_file):
    # Load the data
    books_df = pd.read_csv(books_file)
    movies_df = pd.read_csv(movies_file)

    # Convert embedding strings to float arrays
    books_df['embedding'] = books_df['embedding'].apply(parse_embedding)
    movies_df['embedding'] = movies_df['embedding'].apply(parse_embedding)

    # Initialize lists to store the results
    results_1 = []
    results_2 = []
    all_results = []

    # Iterate through each book record
    for _, book_row in books_df.iterrows():
        book_user_id = book_row['user_id']
        book_asin = book_row['asin']
        book_rating = book_row['rating']
        book_embedding = book_row['embedding']
        book_generated_response = book_row['Generated_Response']

        # Calculate similarity for each movie and store with movie details
        movie_similarities = []
        for _, movie_row in movies_df.iterrows():
            movie_user_id = movie_row['user_id']
            movie_asin = movie_row['asin']
            movie_rating = movie_row['rating']
            movie_embedding = movie_row['embedding']
            movie_generated_response = movie_row['Generated_Response']

            similarity = calculate_similarity(book_embedding, movie_embedding)
            movie_similarities.append((similarity, movie_user_id, movie_asin, movie_rating, movie_generated_response))

            # Append all similarity results to the all_results list
            all_results.append({
                'book_user_id': book_user_id,
                'movie_user_id': movie_user_id,
                'similarity': similarity,
                'book_asin': book_asin,
                'book_rating': book_rating,
                'movie_asin': movie_asin,
                'movie_rating': movie_rating,
                'book_generated_response': book_generated_response,
                'movie_generated_response': movie_generated_response
            })

        # Sort movies by similarity
        movie_similarities.sort(key=lambda x: x[0], reverse=True)

        # Select top 2 and bottom 2 movies
        top_movies = movie_similarities[:2]
        bottom_movies = movie_similarities[-2:]

        # Append top 2 and bottom 2 movies to results for first output
        for similarity, movie_user_id, movie_asin, movie_rating, _ in top_movies + bottom_movies:
            results_1.append({
                'user_id': book_user_id,
                'similarity': similarity,
                'book_asin': book_asin,
                'book_rating': book_rating,
                'movie_asin': movie_asin,
                'movie_rating': movie_rating,
            })

        # Append top 2 and bottom 2 movies to results for second output
        for similarity, movie_user_id, movie_asin, movie_rating, movie_generated_response in top_movies + bottom_movies:
            results_2.append({
                'user_id': book_user_id,
                'similarity': similarity,
                'book_asin': book_asin,
                'book_rating': book_rating,
                'movie_asin': movie_asin,
                'movie_rating': movie_rating,
                'book_generated_response': book_generated_response,
                'movie_generated_response': movie_generated_response
            })

    # Convert the all_results list to a DataFrame
    all_results_df = pd.DataFrame(all_results)

    # Group by 'book_asin' and sort each group by 'similarity' in descending order
    all_results_df = all_results_df.groupby('book_asin', group_keys=False).apply(lambda x: x.sort_values('similarity', ascending=False))


    # Convert the results to DataFrames
    results_df_1 = pd.DataFrame(results_1)
    results_df_2 = pd.DataFrame(results_2)

    # Save the results to CSV files
    results_df_1.to_csv(output_file_1, index=False)
    results_df_2.to_csv(output_file_2, index=False)
    all_results_df.to_csv(all_results_file, index=False)

    print(f"Similarity calculations are complete and saved to {output_file_1}, {output_file_2}, and {all_results_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate similarity between books and movies.')
    parser.add_argument('--books_file', type=str, required=True, help='Path to the books CSV file')
    parser.add_argument('--movies_file', type=str, required=True, help='Path to the movies CSV file')
    parser.add_argument('--output_file_1', type=str, required=True, help='Path to the first output CSV file')
    parser.add_argument('--output_file_2', type=str, required=True, help='Path to the second output CSV file')
    parser.add_argument('--all_results_file', type=str, required=True, help='Path to the file with all similarity results')

    args = parser.parse_args()

    main(args.books_file, args.movies_file, args.output_file_1, args.output_file_2, args.all_results_file)

