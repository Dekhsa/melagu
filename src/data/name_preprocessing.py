import csv
import re
import os

def main():
    # File paths
    base_dir = '/media/dekhsa/DATA/PROJECTS/DS-SongRecommendation/dataset'
    db_nama_path = os.path.join(base_dir, 'DatabaseNama.csv')
    indonesian_name_path = os.path.join(base_dir, 'Indonesian_Name_Dataset.csv')
    name_txt_path = os.path.join(base_dir, 'name.txt')
    output_path = os.path.join(base_dir, 'all_name.csv')

    # Set to store unique names
    nama_set = set()

    def process_name(name_string):
        if not name_string:
            return
        # Replace non-alphabet characters with space to split properly
        cleaned = re.sub(r'[^a-zA-Z\s]', ' ', name_string)
        words = cleaned.split()
        for word in words:
            if len(word) > 0:
                nama_set.add(word.lower())

    # 1. Process DatabaseNama.csv
    try:
        with open(db_nama_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'Nama' in row and row['Nama']:
                    process_name(row['Nama'])
    except Exception as e:
        print(f"Error processing {db_nama_path}: {e}")

    # 2. Process Indonesian_Name_Dataset.csv
    try:
        with open(indonesian_name_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'NAMA' in row and row['NAMA']:
                    process_name(row['NAMA'])
    except Exception as e:
        print(f"Error processing {indonesian_name_path}: {e}")

    # 3. Process name.txt
    try:
        with open(name_txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                process_name(line.strip())
    except Exception as e:
        print(f"Error processing {name_txt_path}: {e}")

    # Convert names to capitalized format and sort them
    nama_list = sorted([name.capitalize() for name in nama_set])

    # Save to all_name.csv
    try:
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['nama'])  # Header
            for name in nama_list:
                writer.writerow([name])
        print(f"Successfully processed and saved {len(nama_list)} unique names to {output_path}")
    except Exception as e:
        print(f"Error saving to {output_path}: {e}")

if __name__ == '__main__':
    main()
