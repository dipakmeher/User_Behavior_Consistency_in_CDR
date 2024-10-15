import gzip
import json
import random

# List of file paths
file_paths = [
    "./data_results/2filtered_Books.jsonl.gz",
    "./data_results/2filtered_CDs_and_Vinyl.jsonl.gz",
    "./data_results/2filtered_Electronic.jsonl.gz",
    "./data_results/2filtered_Grocery_and_Gourmet_Food.jsonl.gz",
    "./data_results/2filtered_Movies_and_TV.jsonl.gz"
]

def extract_user_ids(file_path):
    """Extracts user_id from the given gzipped JSONL file."""
    user_ids = set()  # Use a set to store unique user_ids
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            if 'user_id' in record:
                user_ids.add(record['user_id'])
    return user_ids

def find_overlapping_user_ids(file_paths):
    """Finds the overlapping user_id values across multiple files."""
    # Start with the user_ids from the first file
    overlapping_user_ids = extract_user_ids(file_paths[0])

    # Intersect with user_ids from the other files
    for file_path in file_paths[1:]:
        current_user_ids = extract_user_ids(file_path)
        overlapping_user_ids.intersection_update(current_user_ids)

    return overlapping_user_ids

def select_random_user_ids(user_ids, n=100):
    """Selects `n` random user_id values from a set."""
    if len(user_ids) <= n:
        return list(user_ids)  # If there are less than `n` users, return all
    return random.sample(user_ids, n)

def save_user_ids(user_ids, output_file):
    """Saves the selected user_id values to an output file."""
    with open(output_file, 'w') as f:
        for user_id in user_ids:
            f.write(f"{user_id}\n")

# Main processing
if __name__ == "__main__":
    # Find the overlapping user_ids
    overlapping_user_ids = find_overlapping_user_ids(file_paths)

    # Randomly select 100 user_ids
    selected_user_ids = select_random_user_ids(overlapping_user_ids, n=100)

    # Save the selected user_ids to an output file
    output_file = "./data_results/9selected_user_ids.txt"
    save_user_ids(selected_user_ids, output_file)

    print(f"Selected {len(selected_user_ids)} user_ids saved to {output_file}")

