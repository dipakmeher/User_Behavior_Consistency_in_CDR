import pandas as pd
import argparse

def trim_records(input_file, output_file, records_to_trim):
    # Load the CSV file
    df = pd.read_csv(input_file)

    # Remove the specified number of records from the end
    df_trimmed = df[:-records_to_trim]

    # Save the remaining records to a new CSV file
    df_trimmed.to_csv(output_file, index=False)

    print(f"New file saved as {output_file}")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Trim the specified number of records from the end of a CSV file.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the trimmed output CSV file')
    parser.add_argument('-n', '--num_records', type=int, default=5, help='Number of records to trim from the end (default is 5)')

    args = parser.parse_args()

    # Call the trim_records function with the provided arguments
    trim_records(args.input_file, args.output_file, args.num_records)

