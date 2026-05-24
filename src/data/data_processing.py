"""
data_processing.py — Data loading, preprocessing, and InputExample generation
for the Melagu Song Recommendation System.

Responsibilities:
  1. Load the combined SendTheSong CSV dataset.
  2. Extract unique songs metadata → songs_metadata.csv
  3. Preprocess text messages (slang normalization, emoji→text, cleaning).
  4. Generate sentence-transformers InputExample pairs (anchor=message, positive=song_label).
"""

import csv
import os
import re
import sys
import unicodedata
import logging
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Increase field-size limit for very long messages
csv.field_size_limit(sys.maxsize)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Minimum message length (characters) after cleaning to keep a training row.
MIN_MESSAGE_LENGTH = 10

# Maximum message length (characters) to cap extremely long messages.
MAX_MESSAGE_LENGTH = 512

# ---------------------------------------------------------------------------
# Slang / abbreviation dictionary (Indonesian + English common internet slang)
# ---------------------------------------------------------------------------
SLANG_DICT: Dict[str, str] = {
    # Indonesian
    "gak": "tidak",
    "ga": "tidak",
    "gk": "tidak",
    "ngga": "tidak",
    "nggak": "tidak",
    "gpp": "tidak apa-apa",
    "gapapa": "tidak apa-apa",
    "gapapalah": "tidak apa-apa",
    "yg": "yang",
    "utk": "untuk",
    "dgn": "dengan",
    "dg": "dengan",
    "sm": "sama",
    "sma": "sama",
    "krn": "karena",
    "krna": "karena",
    "tp": "tapi",
    "tpi": "tapi",
    "bgt": "banget",
    "bngt": "banget",
    "bngtt": "banget",
    "bngt": "banget",
    "lg": "lagi",
    "lgi": "lagi",
    "udh": "sudah",
    "udah": "sudah",
    "sdh": "sudah",
    "blm": "belum",
    "blum": "belum",
    "org": "orang",
    "orng": "orang",
    "dr": "dari",
    "dri": "dari",
    "bs": "bisa",
    "bsa": "bisa",
    "aja": "saja",
    "aj": "saja",
    "klo": "kalau",
    "kalo": "kalau",
    "klu": "kalau",
    "emg": "memang",
    "emng": "memang",
    "jg": "juga",
    "jga": "juga",
    "jd": "jadi",
    "jdi": "jadi",
    "td": "tadi",
    "tdi": "tadi",
    "hrs": "harus",
    "msh": "masih",
    "msih": "masih",
    "skrg": "sekarang",
    "skrang": "sekarang",
    "bkn": "bukan",
    "bnr": "benar",
    "sm": "sama",
    "kyk": "kayak",
    "kyak": "kayak",
    "makasih": "terima kasih",
    "makasi": "terima kasih",
    "mksh": "terima kasih",
    "trims": "terima kasih",
    "wkwk": "haha",
    "wkwkwk": "haha",
    "wkwkwkwk": "haha",
    "hehe": "haha",
    "hihi": "haha",
    "kwkw": "haha",
    "moga": "semoga",
    "smoga": "semoga",
    "pgn": "pengen",
    "pengen": "ingin",
    "pngen": "ingin",
    "ak": "aku",
    "gw": "aku",
    "gue": "aku",
    "gua": "aku",
    "km": "kamu",
    "kmu": "kamu",
    "lo": "kamu",
    "lu": "kamu",
    "elu": "kamu",
    "dy": "dia",
    "mrk": "mereka",
    "dmn": "dimana",
    "gmn": "gimana",
    "gmna": "gimana",
    "knp": "kenapa",
    "knpa": "kenapa",
    "brp": "berapa",
    "brpa": "berapa",
    # English
    "u": "you",
    "ur": "your",
    "r": "are",
    "pls": "please",
    "plz": "please",
    "thx": "thanks",
    "thnx": "thanks",
    "ty": "thank you",
    "ily": "i love you",
    "ilysm": "i love you so much",
    "imy": "i miss you",
    "imysm": "i miss you so much",
    "idk": "i don't know",
    "idc": "i don't care",
    "nvm": "never mind",
    "tbh": "to be honest",
    "imo": "in my opinion",
    "btw": "by the way",
    "omg": "oh my god",
    "lol": "haha",
    "lmao": "haha",
    "rn": "right now",
    "smh": "shaking my head",
    "brb": "be right back",
    "jk": "just kidding",
    "bc": "because",
    "cuz": "because",
    "w/": "with",
    "w/o": "without",
    "b4": "before",
    "2nite": "tonight",
    "2day": "today",
    "4ever": "forever",
    "bf": "boyfriend",
    "gf": "girlfriend",
    "dm": "direct message",
    "rly": "really",
    "rlly": "really",
    "srsly": "seriously",
    "abt": "about",
    "wanna": "want to",
    "gonna": "going to",
    "gotta": "got to",
    "kinda": "kind of",
    "sorta": "sort of",
    "dunno": "don't know",
    "lemme": "let me",
    "gimme": "give me",
    # Tagalog common
    "naman": "naman",
    "po": "po",
    "mo": "mo",
    "ko": "ko",
    "sana": "sana",
    "lang": "lang",
    "din": "din",
    "rin": "rin",
    "talaga": "talaga",
    "tlga": "talaga",
    "grbe": "grabe",
}

# Compile slang regex once — match whole words only
_SLANG_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(SLANG_DICT, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Emoji → Text conversion (lightweight, no heavy dependency)
# ---------------------------------------------------------------------------
def _emoji_to_text(text: str) -> str:
    """
    Convert common emoji characters to short English text descriptions.
    Uses a curated dictionary for speed rather than pulling in the full
    `emoji` library.  Remaining unrecognised emoji are stripped.
    """
    emoji_map = {
        "❤️": " love ",
        "❤": " love ",
        "💕": " love ",
        "💗": " love ",
        "💖": " love ",
        "💘": " love ",
        "💝": " love ",
        "💞": " love ",
        "💓": " love ",
        "🥰": " love ",
        "😍": " love ",
        "😘": " kiss ",
        "💋": " kiss ",
        "😊": " happy ",
        "😄": " happy ",
        "😃": " happy ",
        "😁": " happy ",
        "🙂": " happy ",
        "😆": " happy ",
        "🤗": " hug ",
        "😢": " sad ",
        "😭": " crying ",
        "😿": " sad ",
        "🥺": " pleading ",
        "😔": " sad ",
        "😞": " sad ",
        "😩": " sad ",
        "💔": " heartbreak ",
        "😡": " angry ",
        "😠": " angry ",
        "🤬": " angry ",
        "😤": " frustrated ",
        "🙄": " annoyed ",
        "😅": " awkward ",
        "😂": " laughing ",
        "🤣": " laughing ",
        "😎": " cool ",
        "🔥": " fire ",
        "✨": " sparkle ",
        "🌟": " star ",
        "⭐": " star ",
        "🎵": " music ",
        "🎶": " music ",
        "🎧": " music ",
        "🎤": " singing ",
        "🙏": " pray ",
        "👋": " wave ",
        "👍": " good ",
        "👎": " bad ",
        "🤝": " handshake ",
        "🤞": " hoping ",
        "💪": " strong ",
        "😇": " blessed ",
        "🥲": " happy tears ",
        "😶": " speechless ",
        "😐": " neutral ",
        "🫶": " love ",
        "☹️": " sad ",
        "☹": " sad ",
        "🙃": " upside down smile ",
        "🫠": " melting ",
        "😴": " sleepy ",
        "🤧": " sick ",
        "🎂": " birthday ",
        "🎁": " gift ",
        "🌸": " flower ",
        "🌹": " rose ",
        "🌻": " sunflower ",
        "🦋": " butterfly ",
        "🌈": " rainbow ",
        "☀️": " sun ",
        "🌙": " moon ",
        "💤": " sleep ",
        "💫": " dizzy ",
        "🤍": " love ",
        "🖤": " love ",
        "💜": " love ",
        "💙": " love ",
        "💚": " love ",
        "💛": " love ",
        "🧡": " love ",
        "🩷": " love ",
        "🩵": " love ",
    }
    for emoji_char, replacement in emoji_map.items():
        text = text.replace(emoji_char, replacement)
    return text


# ---------------------------------------------------------------------------
# Core text preprocessing
# ---------------------------------------------------------------------------
def preprocess_message(text: str) -> str:
    """
    Full preprocessing pipeline for a single message:
      1. Convert to lowercase.
      2. Convert emoji to text descriptions.
      3. Normalize unicode (NFKD → strip combining marks).
      4. Normalize slang / abbreviations.
      5. Remove URLs, mentions, hashtags.
      6. Remove excessive punctuation and special characters.
      7. Collapse whitespace.
      8. Truncate to MAX_MESSAGE_LENGTH.
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower().strip()

    # 2. Emoji → text
    text = _emoji_to_text(text)

    # 3. Unicode normalisation (e.g. fancy unicode letters → ascii)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))

    # 4. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 5. Remove @mentions and #hashtags
    text = re.sub(r"[@#]\w+", " ", text)

    # 6. Normalise slang (whole-word replacement)
    text = _SLANG_PATTERN.sub(lambda m: SLANG_DICT.get(m.group(0).lower(), m.group(0)), text)

    # 7. Keep only letters, digits, and basic punctuation
    text = re.sub(r"[^a-z0-9\s.,!?'\"-]", " ", text)

    # 8. Collapse repeated punctuation (e.g. "!!!" → "!")
    text = re.sub(r"([!?.])\1+", r"\1", text)

    # 9. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 10. Truncate
    if len(text) > MAX_MESSAGE_LENGTH:
        text = text[:MAX_MESSAGE_LENGTH].rsplit(" ", 1)[0]

    return text


def build_song_label(song_name: str, song_artist: str) -> str:
    """
    Create a composite text label for a song that the sentence-transformer
    can encode on the positive side of the contrastive pair.

    Format: "song_name by song_artist"
    """
    name = (song_name or "").strip()
    artist = (song_artist or "").strip()
    if name and artist:
        return f"{name} by {artist}"
    return name or artist or ""


# ---------------------------------------------------------------------------
# Dataset loading & songs metadata extraction
# ---------------------------------------------------------------------------
def load_dataset(csv_path: str) -> List[Dict[str, str]]:
    """Load the combined CSV dataset into a list of dicts (memory-efficient streaming)."""
    rows: List[Dict[str, str]] = []
    logger.info("Loading dataset from %s ...", csv_path)
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    logger.info("Loaded %d rows.", len(rows))
    return rows


def extract_songs_metadata(
    rows: List[Dict[str, str]], output_path: str
) -> Dict[str, Dict[str, str]]:
    """
    Extract unique songs from the dataset and save to songs_metadata.csv.
    Returns a dict keyed by song_id → {song_name, song_artist, song_image}.
    """
    songs: Dict[str, Dict[str, str]] = {}
    for row in rows:
        sid = row.get("song_id", "").strip()
        if sid and sid not in songs:
            songs[sid] = {
                "song_id": sid,
                "song_name": row.get("song_name", "").strip(),
                "song_artist": row.get("song_artist", "").strip(),
                "song_image": row.get("song_image", "").strip(),
            }

    # Write metadata CSV
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["song_id", "song_name", "song_artist", "song_image"])
        writer.writeheader()
        for song in songs.values():
            writer.writerow(song)

    logger.info("Extracted %d unique songs → %s", len(songs), output_path)
    return songs


# ---------------------------------------------------------------------------
# InputExample generation for Sentence-Transformers
# ---------------------------------------------------------------------------
def build_training_examples(
    rows: List[Dict[str, str]],
) -> List[Tuple[str, str]]:
    """
    Build (anchor, positive) text pairs for MultipleNegativesRankingLoss.

    - anchor  = preprocessed user message
    - positive = song label  ("song_name by song_artist")

    Rows with empty / too-short messages after cleaning are skipped.
    Returns a list of (anchor_text, positive_text) tuples.
    """
    pairs: List[Tuple[str, str]] = []
    skipped = 0

    for row in rows:
        raw_message = row.get("message", "")
        song_name = row.get("song_name", "")
        song_artist = row.get("song_artist", "")

        # Preprocess
        clean_msg = preprocess_message(raw_message)
        song_label = build_song_label(song_name, song_artist)

        # Quality filter
        if len(clean_msg) < MIN_MESSAGE_LENGTH or not song_label:
            skipped += 1
            continue

        pairs.append((clean_msg, song_label))

    logger.info(
        "Built %d training pairs (skipped %d rows due to quality filters).",
        len(pairs),
        skipped,
    )
    return pairs


# ---------------------------------------------------------------------------
# CLI entry-point — run data separation & preview
# ---------------------------------------------------------------------------
def main():
    """
    CLI entry point:
      python -m src.data.data_processing

    Steps:
      1. Load combined dataset.
      2. Extract songs_metadata.csv.
      3. Print sample preprocessed pairs.
    """
    dataset_path = os.path.join("dataset", "combined_sendthesong.csv")
    metadata_output = os.path.join("dataset", "songs_metadata.csv")

    # Step 1: Load
    rows = load_dataset(dataset_path)

    # Step 2: Extract songs metadata
    songs = extract_songs_metadata(rows, metadata_output)

    # Step 3: Build training pairs (just for stats / preview)
    pairs = build_training_examples(rows)

    # Preview
    logger.info("=" * 60)
    logger.info("PREVIEW — first 5 training pairs:")
    logger.info("=" * 60)
    for i, (anchor, positive) in enumerate(pairs[:5]):
        logger.info("[%d] ANCHOR : %s", i, anchor[:120])
        logger.info("[%d] POSITIVE: %s", i, positive)
        logger.info("-" * 40)

    logger.info("Done. Songs metadata: %s (%d songs)", metadata_output, len(songs))
    logger.info("Total training pairs ready: %d", len(pairs))


if __name__ == "__main__":
    main()
