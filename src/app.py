"""
app.py — Streamlit application for the Melagu Song Recommendation System.

Loads the fine-tuned sentence-transformer, pre-computed song embeddings,
and songs metadata. Accepts a user text message, encodes it, performs
cosine similarity search, and displays the Top-10 recommended songs.

Usage:
  streamlit run src/app.py
"""

import csv
import os
import pickle
import sys
from typing import Dict, List

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Local imports — add project root to path for clean imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.data_processing import preprocess_message  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "melagu-e5-finetuned")
EMBEDDINGS_PATH = os.path.join(PROJECT_ROOT, "models", "embeddings", "song_embeddings.npy")
SONG_IDS_PATH = os.path.join(PROJECT_ROOT, "models", "embeddings", "song_ids.pkl")
SONGS_METADATA_PATH = os.path.join(PROJECT_ROOT, "dataset", "songs_metadata.csv")

# E5 query prefix (must match training)
E5_QUERY_PREFIX = "query: "


# ---------------------------------------------------------------------------
# Cached loaders (Streamlit caching for performance)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model Melagu...")
def load_model() -> SentenceTransformer:
    """Load the fine-tuned sentence-transformer model."""
    return SentenceTransformer(MODEL_DIR)


@st.cache_data(show_spinner="Memuat database lagu...")
def load_song_embeddings():
    """Load pre-computed song embedding matrix and ordered song IDs."""
    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
    with open(SONG_IDS_PATH, "rb") as f:
        song_ids = pickle.load(f)
    return embeddings, song_ids


@st.cache_data(show_spinner="Memuat metadata lagu...")
def load_songs_metadata() -> Dict[str, Dict[str, str]]:
    """Load songs metadata into a dict keyed by song_id."""
    songs: Dict[str, Dict[str, str]] = {}
    with open(SONGS_METADATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["song_id"].strip()
            songs[sid] = {
                "song_name": row.get("song_name", "").strip(),
                "song_artist": row.get("song_artist", "").strip(),
                "song_image": row.get("song_image", "").strip(),
            }
    return songs


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------
def recommend_songs(
    message: str,
    model: SentenceTransformer,
    song_embeddings: np.ndarray,
    song_ids: List[str],
    songs_metadata: Dict[str, Dict[str, str]],
    top_k: int = 10,
) -> List[Dict]:
    """
    Encode user message, compute cosine similarity against the song
    embedding database, and return top-K results.
    """
    # Preprocess & encode with E5 query prefix
    clean_msg = preprocess_message(message)
    if not clean_msg:
        return []

    query_text = f"{E5_QUERY_PREFIX}{clean_msg}"
    query_embedding = model.encode(
        [query_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    # Cosine similarity (embeddings are already L2-normalised → dot product)
    similarities = np.dot(song_embeddings, query_embedding.T).flatten()

    # Get top-K indices (descending)
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        sid = song_ids[idx]
        meta = songs_metadata.get(sid, {})
        results.append(
            {
                "rank": len(results) + 1,
                "song_id": sid,
                "song_name": meta.get("song_name", "Unknown"),
                "song_artist": meta.get("song_artist", "Unknown"),
                "song_image": meta.get("song_image", ""),
                "similarity": float(similarities[idx]),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    # --- Page config ---
    st.set_page_config(
        page_title="Melagu — Song Recommendation",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # --- Custom CSS ---
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* Global styling */
        .stApp {
            font-family: 'Inter', sans-serif;
        }

        /* Hero header */
        .hero-title {
            font-size: 3.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.2rem;
            letter-spacing: -0.02em;
        }
        .hero-subtitle {
            text-align: center;
            color: #9ca3af;
            font-size: 1.1rem;
            font-weight: 400;
            margin-bottom: 2rem;
        }

        /* Song card */
        .song-card {
            background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 16px;
            padding: 1rem;
            text-align: center;
            border: 1px solid rgba(102, 126, 234, 0.15);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            height: 100%;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        }
        .song-card:hover {
            transform: translateY(-4px);
            border-color: rgba(102, 126, 234, 0.4);
            box-shadow: 0 12px 40px rgba(102, 126, 234, 0.15);
        }
        .song-rank {
            display: inline-block;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            line-height: 28px;
            font-size: 0.8rem;
            font-weight: 700;
            margin-bottom: 0.6rem;
        }
        .song-image {
            width: 120px;
            height: 120px;
            border-radius: 12px;
            object-fit: cover;
            margin: 0.5rem auto;
            display: block;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }
        .song-name {
            font-size: 0.95rem;
            font-weight: 600;
            color: #e2e8f0;
            margin: 0.5rem 0 0.2rem 0;
            line-height: 1.3;
            overflow: hidden;
            text-overflow: ellipsis;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }
        .song-artist {
            font-size: 0.82rem;
            color: #9ca3af;
            font-weight: 400;
        }
        .song-score {
            font-size: 0.72rem;
            color: #667eea;
            font-weight: 500;
            margin-top: 0.4rem;
        }

        /* Stats badge */
        .stats-container {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin: 1.5rem 0;
        }
        .stat-badge {
            background: rgba(102, 126, 234, 0.1);
            border: 1px solid rgba(102, 126, 234, 0.2);
            border-radius: 12px;
            padding: 0.6rem 1.2rem;
            text-align: center;
        }
        .stat-value {
            font-size: 1.3rem;
            font-weight: 700;
            color: #667eea;
        }
        .stat-label {
            font-size: 0.75rem;
            color: #9ca3af;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Hide Streamlit defaults */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Load resources ---
    model = load_model()
    song_embeddings, song_ids = load_song_embeddings()
    songs_metadata = load_songs_metadata()

    # --- Header ---
    st.markdown('<div class="hero-title">🎵 Melagu</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Ceritakan perasaanmu, kami carikan lagunya.</div>',
        unsafe_allow_html=True,
    )

    # --- Stats ---
    st.markdown(
        f"""
        <div class="stats-container">
            <div class="stat-badge">
                <div class="stat-value">{len(songs_metadata):,}</div>
                <div class="stat-label">Lagu</div>
            </div>
            <div class="stat-badge">
                <div class="stat-value">321,291</div>
                <div class="stat-label">Pesan Terlatih</div>
            </div>
            <div class="stat-badge">
                <div class="stat-value">E5-Base</div>
                <div class="stat-label">Model</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Input area ---
    st.markdown("---")
    user_input = st.text_area(
        "💬 Tulis pesan atau ceritakan perasaanmu:",
        placeholder="Contoh: Aku kangen banget sama dia, rasanya pengen balik lagi ke masa-masa dulu waktu kita masih bareng...",
        height=120,
        key="message_input",
    )

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        search_clicked = st.button(
            "🔍 Cari Lagu",
            use_container_width=True,
            type="primary",
        )

    # --- Results ---
    if search_clicked and user_input.strip():
        with st.spinner("Menganalisis perasaanmu dan mencari lagu yang cocok..."):
            results = recommend_songs(
                message=user_input,
                model=model,
                song_embeddings=song_embeddings,
                song_ids=song_ids,
                songs_metadata=songs_metadata,
                top_k=10,
            )

        if results:
            st.markdown("---")
            st.markdown("### 🎶 Rekomendasi Lagu Untukmu")

            # Render as a 5x2 grid
            for row_start in range(0, len(results), 5):
                cols = st.columns(5)
                for col_idx, song in enumerate(results[row_start : row_start + 5]):
                    with cols[col_idx]:
                        img_url = song["song_image"] if song["song_image"] else "https://via.placeholder.com/120x120.png?text=🎵"
                        st.markdown(
                            f"""
                            <div class="song-card">
                                <div class="song-rank">{song['rank']}</div>
                                <img class="song-image" src="{img_url}" alt="{song['song_name']}" loading="lazy" onerror="this.src='https://via.placeholder.com/120x120.png?text=🎵'">
                                <div class="song-name">{song['song_name']}</div>
                                <div class="song-artist">{song['song_artist']}</div>
                                <div class="song-score">Score: {song['similarity']:.4f}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.warning("Pesan terlalu pendek atau tidak valid. Coba tulis lebih detail.")

    elif search_clicked:
        st.warning("Tulis pesan terlebih dahulu sebelum mencari lagu.")

    # --- Footer ---
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #6b7280; font-size: 0.8rem; padding: 1rem 0;">
            Melagu — Dibangun dengan Sentence-Transformers & multilingual-e5-base<br>
            Dataset: 321.291 pesan dari SendTheSong
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
