"""
train.py — Fine-tune intfloat/multilingual-e5-base with MultipleNegativesRankingLoss
for the Melagu Song Recommendation System.

Integrates:
  - MLflow tracking (configured for Dagshub)
  - CLI hyperparameter tuning via argparse
  - Song embedding generation & persistence (numpy + pickle)

Usage:
  python -m src.train --epochs 5 --batch_size 64 --learning_rate 2e-5 --warmup_steps 500

Environment variables (for Dagshub MLflow):
  MLFLOW_TRACKING_URI   — e.g. https://dagshub.com/<user>/<repo>.mlflow
  MLFLOW_TRACKING_USERNAME — Dagshub username
  MLFLOW_TRACKING_PASSWORD — Dagshub token
"""

import argparse
import csv
import gc
import logging
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import pickle
import sys
import time
from typing import List, Tuple

import numpy as np
import torch
from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    losses,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from src.data.data_processing import (
    build_song_label,
    build_training_examples,
    extract_songs_metadata,
    load_dataset,
    preprocess_message,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_PATH = os.path.join("dataset", "combined_sendthesong.csv")
SONGS_METADATA_PATH = os.path.join("dataset", "songs_metadata.csv")
MODEL_OUTPUT_DIR = os.path.join("models", "melagu-e5-finetuned")
EMBEDDINGS_DIR = os.path.join("models", "embeddings")
SONG_EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "song_embeddings.npy")
SONG_IDS_PATH = os.path.join(EMBEDDINGS_DIR, "song_ids.pkl")

# ---------------------------------------------------------------------------
# E5 model prefix — required by intfloat/multilingual-e5-base
# ---------------------------------------------------------------------------
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "


# ---------------------------------------------------------------------------
# InputExample builder
# ---------------------------------------------------------------------------
def create_input_examples(pairs: List[Tuple[str, str]]) -> List[InputExample]:
    """
    Convert (anchor, positive) text tuples into sentence-transformers
    InputExample objects.  The E5 model family requires "query: " / "passage: "
    prefixes for optimal performance.
    """
    examples = []
    for anchor, positive in pairs:
        examples.append(
            InputExample(
                texts=[
                    f"{E5_QUERY_PREFIX}{anchor}",
                    f"{E5_PASSAGE_PREFIX}{positive}",
                ]
            )
        )
    return examples


# ---------------------------------------------------------------------------
# Song embedding generation
# ---------------------------------------------------------------------------
def generate_song_embeddings(
    model: SentenceTransformer,
    songs_metadata_path: str,
    output_embeddings_path: str,
    output_ids_path: str,
    batch_size: int = 128,
) -> None:
    """
    Encode all unique song labels into dense vectors and persist them.

    Saves:
      - song_embeddings.npy — (N, D) float32 matrix
      - song_ids.pkl — ordered list of song_id strings matching the matrix rows
    """
    logger.info("Generating song embeddings from %s ...", songs_metadata_path)

    song_ids: List[str] = []
    song_labels: List[str] = []

    with open(songs_metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["song_id"].strip()
            label = build_song_label(row["song_name"], row["song_artist"])
            if sid and label:
                song_ids.append(sid)
                # E5 passage prefix for song labels
                song_labels.append(f"{E5_PASSAGE_PREFIX}{label}")

    logger.info("Encoding %d unique songs ...", len(song_labels))
    embeddings = model.encode(
        song_labels,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # Persist
    os.makedirs(os.path.dirname(output_embeddings_path), exist_ok=True)
    np.save(output_embeddings_path, embeddings.astype(np.float32))

    with open(output_ids_path, "wb") as f:
        pickle.dump(song_ids, f)

    logger.info(
        "Saved song embeddings → %s  shape=%s",
        output_embeddings_path,
        embeddings.shape,
    )
    logger.info("Saved song IDs → %s  count=%d", output_ids_path, len(song_ids))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> None:
    """Main training loop with MLflow tracking."""

    # ------------------------------------------------------------------
    # 1. MLflow / Dagshub setup
    # ------------------------------------------------------------------
    try:
        import mlflow

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info("MLflow tracking URI: %s", tracking_uri)
        else:
            # Fall back to local mlruns directory
            mlflow.set_tracking_uri("file:./mlruns")
            logger.info("MLflow tracking URI: local ./mlruns")

        mlflow.set_experiment(args.experiment_name)
        mlflow_available = True
    except ImportError:
        logger.warning("mlflow not installed — skipping experiment tracking.")
        mlflow_available = False

    # ------------------------------------------------------------------
    # 2. Load & preprocess data
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1: Loading and preprocessing dataset")
    logger.info("=" * 60)

    rows = load_dataset(DATASET_PATH)
    songs = extract_songs_metadata(rows, SONGS_METADATA_PATH)
    pairs = build_training_examples(rows)

    # Free raw rows from memory — we only need pairs now
    del rows
    gc.collect()

    logger.info("Creating InputExamples ...")
    train_examples = create_input_examples(pairs)
    del pairs
    gc.collect()
    logger.info("Total InputExamples: %d", len(train_examples))

    # ------------------------------------------------------------------
    # 3. Load pre-trained model
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 2: Loading pre-trained model: %s", args.model_name)
    logger.info("=" * 60)

    model = SentenceTransformer(args.model_name)

    # ------------------------------------------------------------------
    # 4. DataLoader & Loss
    # ------------------------------------------------------------------
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )

    # MultipleNegativesRankingLoss: uses in-batch negatives
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Calculate total training steps
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(args.warmup_steps, total_steps // 5)

    logger.info("Training config:")
    logger.info("  Model           : %s", args.model_name)
    logger.info("  Epochs          : %d", args.epochs)
    logger.info("  Batch size      : %d", args.batch_size)
    logger.info("  Learning rate   : %.2e", args.learning_rate)
    logger.info("  Warmup steps    : %d", warmup_steps)
    logger.info("  Steps/epoch     : %d", steps_per_epoch)
    logger.info("  Total steps     : %d", total_steps)
    logger.info("  Unique songs    : %d", len(songs))
    logger.info("  Training pairs  : %d", len(train_examples))

    # ------------------------------------------------------------------
    # 5. MLflow run & Training
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 3: Starting training")
    logger.info("=" * 60)

    if mlflow_available:
        with mlflow.start_run(run_name=args.run_name):
            # Log hyperparameters
            mlflow.log_param("model_name", args.model_name)
            mlflow.log_param("epochs", args.epochs)
            mlflow.log_param("batch_size", args.batch_size)
            mlflow.log_param("learning_rate", args.learning_rate)
            mlflow.log_param("warmup_steps", warmup_steps)
            mlflow.log_param("total_training_pairs", len(train_examples))
            mlflow.log_param("unique_songs", len(songs))
            mlflow.log_param("loss_function", "MultipleNegativesRankingLoss")

            start_time = time.time()

            # Train the model
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=args.epochs,
                warmup_steps=warmup_steps,
                output_path=MODEL_OUTPUT_DIR,
                show_progress_bar=True,
                optimizer_params={"lr": args.learning_rate},
                use_amp=args.use_amp,
            )

            training_time = time.time() - start_time

            # Log training metrics
            mlflow.log_metric("training_time_seconds", training_time)
            mlflow.log_metric("training_time_minutes", training_time / 60)
            mlflow.log_metric("steps_per_epoch", steps_per_epoch)
            mlflow.log_metric("total_steps", total_steps)

            logger.info("Training complete in %.1f minutes.", training_time / 60)

            # ----------------------------------------------------------
            # 6. Generate & save song embeddings
            # ----------------------------------------------------------
            logger.info("=" * 60)
            logger.info("STEP 4: Generating song embeddings")
            logger.info("=" * 60)

            # Reload the best saved model
            model = SentenceTransformer(MODEL_OUTPUT_DIR)

            generate_song_embeddings(
                model=model,
                songs_metadata_path=SONGS_METADATA_PATH,
                output_embeddings_path=SONG_EMBEDDINGS_PATH,
                output_ids_path=SONG_IDS_PATH,
                batch_size=args.batch_size,
            )

            # Log artifacts to MLflow
            mlflow.log_artifact(SONGS_METADATA_PATH)
            mlflow.log_artifact(SONG_EMBEDDINGS_PATH)
            mlflow.log_artifact(SONG_IDS_PATH)

            # Log embedding shape
            emb = np.load(SONG_EMBEDDINGS_PATH)
            mlflow.log_metric("embedding_dim", emb.shape[1])
            mlflow.log_metric("num_song_embeddings", emb.shape[0])

            logger.info("All artifacts logged to MLflow.")

    else:
        # Train without MLflow
        start_time = time.time()

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=args.epochs,
            warmup_steps=warmup_steps,
            output_path=MODEL_OUTPUT_DIR,
            show_progress_bar=True,
            optimizer_params={"lr": args.learning_rate},
            use_amp=args.use_amp,
        )

        training_time = time.time() - start_time
        logger.info("Training complete in %.1f minutes.", training_time / 60)

        # Generate embeddings
        model = SentenceTransformer(MODEL_OUTPUT_DIR)
        generate_song_embeddings(
            model=model,
            songs_metadata_path=SONGS_METADATA_PATH,
            output_embeddings_path=SONG_EMBEDDINGS_PATH,
            output_ids_path=SONG_IDS_PATH,
            batch_size=args.batch_size,
        )

    logger.info("=" * 60)
    logger.info("DONE — Model saved to: %s", MODEL_OUTPUT_DIR)
    logger.info("DONE — Embeddings saved to: %s", EMBEDDINGS_DIR)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Melagu — Train Song Recommendation Model (Sentence-Transformers + E5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="intfloat/multilingual-e5-base",
        help="HuggingFace model identifier for the base sentence-transformer.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Training & encoding batch size.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Peak learning rate for AdamW optimiser.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of linear warmup steps.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="melagu-song-recommendation",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="MLflow run name (auto-generated if omitted).",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=False,
        help="Enable Automatic Mixed Precision (AMP) for faster training on GPU.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.run_name is None:
        args.run_name = f"e{args.epochs}_bs{args.batch_size}_lr{args.learning_rate}"
    train(args)
