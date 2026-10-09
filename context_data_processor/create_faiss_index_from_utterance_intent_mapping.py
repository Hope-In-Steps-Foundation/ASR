
import json
import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(r"C:\asr\utterance_intent_mapping.csv")
OUTPUT_DIR = Path(r"C:\asr\index")

INDEX_FILE = OUTPUT_DIR / "utterance.index"
METADATA_FILE = OUTPUT_DIR / "metadata.json"

# A compact general-purpose sentence embedding model.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 64


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(value):
    """Normalize whitespace and remove punctuation."""
    if pd.isna(value):
        return ""

    value = re.sub(r"\s+", " ", str(value)).strip()
    value = re.sub(r"[^a-zA-Z0-9 ]", "", value)

    return value.strip()


# ============================================================
# BUILD INDEX
# ============================================================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load utterance mappings.
    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
    )

    df.columns = df.columns.str.strip()

    required_columns = ["utterance", "new_intent"]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Clean text and remove rows without required data.
    df["utterance"] = df["utterance"].map(clean_text)
    df["new_intent"] = df["new_intent"].map(clean_text)

    df = df[
        (df["utterance"] != "")
        & (df["new_intent"] != "")
    ].copy()

    if df.empty:
        raise ValueError("No valid utterance mappings found.")

    # Preserve the IVR intent if available.
    if "ivr_intent" not in df.columns:
        df["ivr_intent"] = ""
    else:
        df["ivr_intent"] = df["ivr_intent"].map(clean_text)

    # Keep distinct utterance-intent pairs.
    df = df.drop_duplicates(
        subset=["utterance", "new_intent", "ivr_intent"]
    ).reset_index(drop=True)

    utterances = df["utterance"].tolist()

    print(f"Mappings to index: {len(df)}")
    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    # Encode utterances into normalized embeddings.
    embeddings = model.encode(
        utterances,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Normalized vectors + inner product = cosine similarity.
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save the FAISS index.
    faiss.write_index(index, str(INDEX_FILE))

    # Save row metadata. Position i corresponds to FAISS ID i.
    metadata = {
        "model_name": MODEL_NAME,
        "vector_dimension": dimension,
        "distance_metric": "cosine_similarity",
        "records": [
            {
                "id": i,
                "utterance": row["utterance"],
                "ivr_intent": row["ivr_intent"],
                "new_intent": row["new_intent"],
            }
            for i, row in df.iterrows()
        ],
    }

    with METADATA_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nFAISS index created successfully.")
    print(f"Vectors indexed: {index.ntotal}")
    print(f"Embedding dimensions: {dimension}")
    print(f"FAISS index: {INDEX_FILE}")
    print(f"Metadata: {METADATA_FILE}")


if __name__ == "__main__":
    main()
