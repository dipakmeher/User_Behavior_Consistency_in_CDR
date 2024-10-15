import argparse
import gzip
import json
import csv

def main(meta_file, data_file, output_file, log_file, selected_columns):
    # Load metadata
    metadata_dict = {}
    with gzip.open(meta_file, 'rt', encoding='utf-8') as fp:
        meta_count = 0
        for line in fp:
            meta_data = json.loads(line.strip())
            if 'parent_asin' in meta_data:
                metadata_dict[meta_data['parent_asin']] = meta_data
            meta_count += 1

    # Merge actual data with metadata
    merged_data = []
    with gzip.open(data_file, 'rt', encoding='utf-8') as fp:
        data_count = 0
        for line in fp:
            actual_data = json.loads(line.strip())
            if 'parent_asin' in actual_data:
                parent_asin = actual_data['parent_asin']
                if parent_asin in metadata_dict:
                    merged_record = {**actual_data, **metadata_dict[parent_asin]}

                    # Remove 'images' from actual data and keep from metadata only
                    if 'images' in actual_data:
                        del actual_data['images']

                    # Keep only selected columns
                    filtered_record = {key: merged_record[key] for key in selected_columns if key in merged_record}
                    merged_data.append(filtered_record)
            data_count += 1

    # Write merged data to a new file
    with gzip.open(output_file, 'wt', encoding='utf-8') as fp:
        for record in merged_data:
            fp.write(json.dumps(record) + '\n')

    # Logging the print output in a log file
    log_data = [
        ["File", "Record Count"],
        [meta_file, meta_count],
        [data_file, data_count],
        [output_file, len(merged_data)]
    ]

    with open(log_file, 'w', newline='', encoding='utf-8') as log_csv:
        log_writer = csv.writer(log_csv)
        log_writer.writerows(log_data)

    print(f"Number of records in {meta_file}: {meta_count}")
    print(f"Number of records in {data_file}: {data_count}")
    print(f"Number of merged records written to {output_file}: {len(merged_data)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge metadata with actual data and save selected columns to a new file.")
    parser.add_argument("--meta_file", type=str, required=True, help="Path to the metadata file (gzip format).")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the actual data file (gzip format).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the merged output file (gzip format).")
    parser.add_argument("--log_file", type=str, required=True, help="Path to save the log file.")
    parser.add_argument("--selected_columns", nargs='+', required=True, help="List of selected columns to save in the output files.")

    args = parser.parse_args()

    main(args.meta_file, args.data_file, args.output_file, args.log_file, args.selected_columns)
