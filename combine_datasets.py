import pandas as pd
import glob
import os

folder = 'dataset/messy-sendthesong'
files = glob.glob(os.path.join(folder, '*.csv'))

df_list = []
for f in files:
    print(f"Reading {f}...")
    try:
        df = pd.read_csv(f, dtype=str) # Read as string to avoid mixed types
        df_list.append(df)
    except Exception as e:
        print(f"Error reading {f}: {e}")

if df_list:
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"Total rows before deduplication: {len(combined_df)}")
    combined_df.drop_duplicates(subset=['_id'], inplace=True)
    print(f"Total rows after deduplication: {len(combined_df)}")
    
    output_path = 'dataset/combined_sendthesong.csv'
    combined_df.to_csv(output_path, index=False)
    print(f"Saved combined data to {output_path}")
else:
    print("No data found.")

