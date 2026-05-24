import requests
import pandas as pd
import time
import json

# --- KONFIGURASI ---
API_URL = "https://api.sendthesong.xyz/api/posts"
OUTPUT_FILENAME = "hasil_pencarian_nama_tergabung.csv"
LIMIT_PER_PAGE = 50
MAX_PAGES_PER_QUERY = 10 # Batas aman agar tidak terjebak pada nama yg hasilnya ribuan

# 1. Daftar Pencarian yang Diperluas
# Menambahkan beberapa variasi umum untuk cakupan maksimal
SEARCH_QUERIES = [
    'muhammad', 'aditya', 'adit', 'rizky', 'rizki', 'fauzan', 'reza',
    'kevin', 'arya', 'fajar', 'dimas', 'gilang', 'putri', 'anisa',
    'anissa', 'amanda', 'sarah', 'nadia', 'dinda', 'cindy', 'sindy',
    'lestari', 'nabila', 'ayu'
]

# Tambahkan nama dari file CSV
try:
    df_names = pd.read_csv('/content/all_name.csv')
    # Assuming the names are in the first column of the CSV
    new_names = df_names.iloc[:, 0].dropna().astype(str).tolist()
    print(f"[+] Berhasil memuat {len(new_names)} nama dari /content/all_name.csv")
    # Convert to set for deduplication and then back to list
    SEARCH_QUERIES = list(set(SEARCH_QUERIES + new_names))
    print(f"[+] Total SEARCH_QUERIES setelah penambahan: {len(SEARCH_QUERIES)}")
except FileNotFoundError:
    print("[!] File /content/all_name.csv tidak ditemukan. Melanjutkan dengan daftar pencarian yang ada.")
except Exception as e:
    print(f"[!] Error saat memuat atau memproses /content/all_name.csv: {e}")

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    'Referer': 'https.sendthesong.xyz/'
}

# --- PROSES SCRAPING ---
all_songs_data = []
total_queries = len(SEARCH_QUERIES)

# 2. Loop Luar: Iterasi untuk setiap nama
for index, query in enumerate(SEARCH_QUERIES):
    print(f"\n{'='*50}")
    print(f"[*] Memulai pencarian untuk: '{query}' ({index + 1}/{total_queries})")
    print(f"{'='*50}")

    current_page = 1

    # 3. Loop Dalam: Pagiansi untuk nama saat ini
    while current_page <= MAX_PAGES_PER_QUERY:
        params = {
            'q': query,
            'page': current_page,
            'limit': LIMIT_PER_PAGE
        }

        print(f"  -> Mencari '{query}', halaman {current_page}...")

        try:
            response = requests.get(API_URL, params=params, headers=HEADERS)

            if response.status_code == 200:
                data = response.json()

                # ASUMSI PENTING: ID unik lagu ada di key '_id'.
                # Ganti '_id' jika nama key-nya berbeda (misal: 'song_id', 'track_id')
                new_songs = data.get('data', [])

                if not new_songs:
                    print(f"  -- Tidak ada hasil lagi untuk '{query}'. Lanjut ke nama berikutnya.")
                    break

                all_songs_data.extend(new_songs)
                current_page += 1
                time.sleep(1.2) # Jeda sedikit lebih lama karena ini proses yang intensif
            else:
                print(f"  [!] Gagal di halaman {current_page} untuk '{query}'. Status: {response.status_code}. Lanjut ke nama berikutnya.")
                break
        except requests.exceptions.RequestException as e:
            print(f"  [!] Error koneksi untuk '{query}': {e}. Lanjut ke nama berikutnya.")
            break

print(f"\n{'='*50}")
print(f"[+] PROSES PENGAMBILAN DATA SELESAI")
print(f"Total data mentah (termasuk duplikat): {len(all_songs_data)}")
print(f"{'='*50}\n")


# --- PROSES DEDUPLIKASI & EXPORT ---
if all_songs_data:
    print("[*] Memulai proses konversi dan deduplikasi...")
    df = pd.DataFrame(all_songs_data)

    # 4. Deduplikasi Cerdas
    # Menghapus baris yang duplikat berdasarkan kolom '_id'.
    # Ganti '_id' dengan kolom unik dari datamu jika berbeda.
    try:
        initial_rows = len(df)
        df.drop_duplicates(subset=['_id'], keep='first', inplace=True)
        final_rows = len(df)

        print(f"[+] Deduplikasi berhasil. Dihapus {initial_rows - final_rows} data duplikat.")
        print(f"[+] Total data unik: {final_rows}")

        # 5. Ekspor Final
        df.to_csv(OUTPUT_FILENAME, index=False, encoding='utf-8')
        print(f"\n[SUCCESS] Data unik telah berhasil diekspor ke file: '{OUTPUT_FILENAME}'")
    except KeyError:
        print("\n[!!!] ERROR: Kolom '_id' tidak ditemukan untuk deduplikasi.")
        print("    Silakan periksa output JSON-mu, temukan nama kolom ID uniknya, dan ubah di bagian df.drop_duplicates(subset=['_id']).")
        print("    Menyimpan data mentah tanpa deduplikasi sebagai gantinya...")
        df.to_csv(f"MENTAH_{OUTPUT_FILENAME}", index=False, encoding='utf-8')
else:
    print("[!] Tidak ada data untuk diekspor.")