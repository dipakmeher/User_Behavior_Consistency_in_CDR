import pandas as pd
import argparse

def process_correlation_summary(input_file, output_file):
    # Load the CSV file
    df = pd.read_csv(input_file)

    # Calculate the average correlation score and count of each score (-1, 0, 1) per user
    result_df = df.groupby('user_id').agg(
        avg_correlation_score=('correlation_score', 'mean'),
        count_1s=('correlation_score', lambda x: (x == 1).sum()),
        count_0s=('correlation_score', lambda x: (x == 0).sum()),
        count_minus1s=('correlation_score', lambda x: (x == -1).sum())
    ).reset_index()

    # Save the result to a new CSV file
    result_df.to_csv(output_file, index=False)

    print(f"User correlation summary saved as {output_file}")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Calculate user correlation summary.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output CSV file')

    args = parser.parse_args()

    # Call the process_correlation_summary function with the provided arguments
    process_correlation_summary(args.input_file, args.output_file)

