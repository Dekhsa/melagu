import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import re

# --- Model Classes ---
class MessageTower(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config["text_embedding_dim"], padding_idx=0)
        self.gru = nn.GRU(
            config["text_embedding_dim"],
            config["hidden_dim"],
            num_layers=config["n_layers"],
            dropout=config["dropout"],
            batch_first=True
        )
        self.fc = nn.Linear(config["hidden_dim"], config["output_embedding_dim"])

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)
        hidden = hidden[-1]
        return F.normalize(self.fc(hidden), p=2, dim=1)

class SongTower(nn.Module):
    def __init__(self, num_songs, config):
        super().__init__()
        self.embedding = nn.Embedding(num_songs, config["output_embedding_dim"])

    def forward(self, x):
        return F.normalize(self.embedding(x), p=2, dim=1)

# --- Utility Functions ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def vectorize_message(text, vocab, max_len):
    tokens = [vocab.get(word, vocab['<UNK>']) for word in text.split()]
    if len(tokens) < max_len:
        tokens += [vocab['<PAD>']] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

@st.cache_data
def load_data(csv_path):
    return pd.read_csv(csv_path)

@st.cache_resource
def load_model_and_assets(model_path, config):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    vocab = checkpoint['vocab']
    songint_to_name = checkpoint['songint_to_name']
    songint_to_original_id = checkpoint['songint_to_original_id']
    vocab_size = len(vocab)
    num_songs = len(songint_to_name)
    message_tower = MessageTower(vocab_size, config)
    song_tower = SongTower(num_songs, config)
    message_tower.load_state_dict(checkpoint['message_tower_state_dict'])
    song_tower.load_state_dict(checkpoint['song_tower_state_dict'])
    message_tower.eval()
    song_tower.eval()
    # Build song index
    with torch.no_grad():
        all_song_ids = torch.arange(num_songs)
        song_index = song_tower(all_song_ids)
    return message_tower, song_index, vocab, songint_to_name, songint_to_original_id

def recommend_songs_streamlit(message, message_tower, song_index, inv_song_map, vocab, config, top_k=10):
    cleaned = clean_text(message)
    vec = vectorize_message(cleaned, vocab, config["max_len"])
    with torch.no_grad():
        query_emb = message_tower(vec)
        similarities = F.cosine_similarity(query_emb, song_index)
        top_results = torch.topk(similarities, k=top_k)
    recs = []
    for score, idx in zip(top_results.values, top_results.indices):
        recs.append({
            "song": inv_song_map[idx.item()],
            "similarity": float(score.item())
        })
    return recs

# --- CONFIG (harus sama dengan training) ---
CONFIG = {
    "text_embedding_dim": 150,
    "hidden_dim": 300,
    "n_layers": 2,
    "dropout": 0.5,
    "output_embedding_dim": 64,
    "batch_size": 128,
    "learning_rate": 0.001,
    "epochs": 25,
    "max_len": 50,
    "margin": 0.2,
    "min_song_freq": 5,
}

st.title('Melagu')
songs_df = load_data('all_songs_data_50k.csv')
message_tower, song_index, vocab, inv_song_map = load_model_and_assets('two_tower_songrec_model.pth', CONFIG)

user_input = st.text_area('Masukkan pesan untuk mendapatkan rekomendasi lagu:')
if st.button('Rekomendasikan Lagu'):
    if user_input:
        recs = recommend_songs_streamlit(user_input, message_tower, song_index, inv_song_map, vocab, CONFIG, top_k=10)
        st.write('Rekomendasi Lagu:')
        st.dataframe(pd.DataFrame(recs))
    else:
        st.warning('Masukkan pesan atau teks terlebih dahulu.')
