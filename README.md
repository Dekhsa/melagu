# Melagu — Song Recommendation System

> Ceritakan perasaanmu, kami carikan lagunya.

Sistem rekomendasi lagu berbasis teks yang menggunakan **Sentence-Transformers** (`intfloat/multilingual-e5-base`) dan **MultipleNegativesRankingLoss** untuk memahami pesan emosional pengguna dan merekomendasikan lagu yang relevan.

**Total dataset:** 321.291 baris data dari SendTheSong  
**Unique songs:** 7.327 lagu

---

## 📁 Struktur Proyek

```
DS-SongRecommendation/
├── dataset/
│   ├── combined_sendthesong.csv        # Dataset gabungan (321k rows)
│   ├── songs_metadata.csv              # Metadata lagu unik (auto-generated)
│   └── messy-sendthesong/              # Raw data files
├── models/
│   ├── melagu-e5-finetuned/            # Fine-tuned model (auto-generated)
│   └── embeddings/
│       ├── song_embeddings.npy         # Dense song vectors (auto-generated)
│       └── song_ids.pkl                # Ordered song IDs (auto-generated)
├── src/
│   ├── data/
│   │   ├── data_processing.py          # Data loading, preprocessing, InputExamples
│   │   ├── create_dataset.py           # Dataset combiner utility
│   │   └── name_preprocessing.py       # Name preprocessing utility
│   ├── train.py                        # Model training + MLflow tracking
│   └── app.py                          # Streamlit deployment
├── configs/
│   └── config.yaml
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Buat Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# atau: venv\Scripts\activate  # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup Dagshub / MLflow (Opsional)

Buat file `.env` atau export environment variables:

```bash
export MLFLOW_TRACKING_URI="https://dagshub.com/<username>/<repo-name>.mlflow"
export MLFLOW_TRACKING_USERNAME="<dagshub-username>"
export MLFLOW_TRACKING_PASSWORD="<dagshub-access-token>"
```

> Jika tidak dikonfigurasi, MLflow akan otomatis menggunakan direktori lokal `./mlruns`.

---

## 🚀 Cara Menjalankan

### Step 1: Data Separation — Ekstrak Metadata Lagu

```bash
python -m src.data.data_processing
```

Output:
- `dataset/songs_metadata.csv` — Metadata 7.327 lagu unik

### Step 2: Training Model

**Default config:**
```bash
python -m src.train
```

**Custom hyperparameters via CLI:**
```bash
python -m src.train \
  --epochs 5 \
  --batch_size 64 \
  --learning_rate 2e-5 \
  --warmup_steps 500 \
  --experiment_name "melagu-song-recommendation" \
  --run_name "experiment-v1"
```

**Dengan Mixed Precision (GPU):**
```bash
python -m src.train --epochs 3 --batch_size 128 --use_amp
```

Output:
- `models/melagu-e5-finetuned/` — Fine-tuned sentence-transformer model
- `models/embeddings/song_embeddings.npy` — Dense song vectors
- `models/embeddings/song_ids.pkl` — Ordered song ID list

### Step 3: Launch Streamlit App

```bash
streamlit run src/app.py
```

---

## 🎛️ Hyperparameter Tuning (CLI Arguments)

| Argument | Type | Default | Deskripsi |
|---|---|---|---|
| `--model_name` | str | `intfloat/multilingual-e5-base` | Base model dari HuggingFace |
| `--epochs` | int | `3` | Jumlah epoch training |
| `--batch_size` | int | `64` | Ukuran batch untuk training & encoding |
| `--learning_rate` | float | `2e-5` | Peak learning rate (AdamW) |
| `--warmup_steps` | int | `500` | Jumlah step linear warmup |
| `--experiment_name` | str | `melagu-song-recommendation` | Nama experiment di MLflow |
| `--run_name` | str | auto-generated | Nama run di MLflow |
| `--use_amp` | flag | `False` | Aktifkan Mixed Precision (AMP) |

**Contoh tuning:**
```bash
# Experiment 1: Conservative
python -m src.train --epochs 2 --batch_size 32 --learning_rate 1e-5 --warmup_steps 200

# Experiment 2: Aggressive  
python -m src.train --epochs 5 --batch_size 128 --learning_rate 5e-5 --warmup_steps 1000 --use_amp

# Experiment 3: Long training
python -m src.train --epochs 10 --batch_size 64 --learning_rate 2e-5 --warmup_steps 500
```

---

## 📊 MLflow Tracking

Semua hyperparameter dan metrik secara otomatis dicatat ke MLflow:

**Parameters yang di-log:**
- `model_name`, `epochs`, `batch_size`, `learning_rate`, `warmup_steps`
- `total_training_pairs`, `unique_songs`, `loss_function`

**Metrics yang di-log:**
- `training_time_seconds`, `training_time_minutes`
- `steps_per_epoch`, `total_steps`
- `embedding_dim`, `num_song_embeddings`

**Artifacts yang di-log:**
- `songs_metadata.csv`, `song_embeddings.npy`, `song_ids.pkl`

Akses dashboard MLflow:
```bash
# Lokal
mlflow ui --port 5000

# Dagshub
# Buka: https://dagshub.com/<username>/<repo-name>.mlflow
```

---

## 🧠 Arsitektur

```
User Message → Preprocessing → E5 Encoder → Query Vector
                                                    ↓
                                            Cosine Similarity
                                                    ↓
Song Database → E5 Encoder → Song Vectors → Top-10 Results
```

1. **Preprocessing**: Slang normalization, emoji→text, unicode cleanup
2. **Encoding**: `intfloat/multilingual-e5-base` (fine-tuned) dengan `query:` / `passage:` prefix
3. **Retrieval**: Dot product (normalized vectors) → cosine similarity → top-K ranking
4. **Display**: Streamlit grid dengan album art, nama lagu, artis, dan skor similarity
