import os
import glob
import csv
import sys

# Increase CSV field size limit to handle potentially very long messages
csv.field_size_limit(sys.maxsize)

def combine_csv_datasets(input_folder, output_file):
    print(f"Searching for CSV files in: {input_folder}")
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    if not csv_files:
        print("No CSV files found in the specified folder.")
        return
    
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f" - {os.path.basename(f)} ({os.path.getsize(f) / (1024*1024):.2f} MB)")
        
    seen_ids = set()
    total_rows_read = 0
    total_rows_written = 0
    header = None
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"\nCombining files into {output_file}...")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        
        for file_path in csv_files:
            file_name = os.path.basename(file_path)
            print(f"Processing {file_name}...")
            
            rows_in_file = 0
            duplicates_in_file = 0
            
            with open(file_path, 'r', newline='', encoding='utf-8', errors='replace') as infile:
                reader = csv.reader(infile)
                
                try:
                    file_header = next(reader)
                except StopIteration:
                    print(f"  Empty file: {file_name}")
                    continue
                
                # Use the header from the first non-empty file
                if header is None:
                    header = file_header
                    writer.writerow(header)
                    print(f"  Header detected: {header}")
                
                # Find index of '_id' column, default to 0 if not found
                try:
                    id_index = file_header.index('_id')
                except ValueError:
                    id_index = 0
                    print(f"  Warning: '_id' column not found in {file_name}. Using column index 0 as ID.")
                
                for row in reader:
                    if not row:
                        continue
                    rows_in_file += 1
                    total_rows_read += 1
                    
                    # Extract ID, use entire row if index is out of bounds
                    row_id = row[id_index] if len(row) > id_index else "".join(row)
                    
                    if row_id not in seen_ids:
                        seen_ids.add(row_id)
                        writer.writerow(row)
                        total_rows_written += 1
                    else:
                        duplicates_in_file += 1
            
            print(f"  Done. Read {rows_in_file} rows. Found {duplicates_in_file} duplicate(s).")
            
    print("\n--- Summary ---")
    print(f"Total CSV files processed: {len(csv_files)}")
    print(f"Total rows read: {total_rows_read}")
    print(f"Total unique rows written: {total_rows_written}")
    print(f"Total duplicates removed: {total_rows_read - total_rows_written}")
    print(f"Combined file saved at: {output_file} ({os.path.getsize(output_file) / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    input_folder = os.path.join("dataset", "messy-sendthesong")
    output_file = os.path.join("dataset", "combined_sendthesong.csv")
    combine_csv_datasets(input_folder, output_file)
