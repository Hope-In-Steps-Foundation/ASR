
import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

# Utterance index
UTTERANCE_INDEX_FILE = Path(r"C:\asr\index\utterance.index")
UTTERANCE_METADATA_FILE = Path(r"C:\asr\index\metadata.json")

# Description index
DESCRIPTION_INDEX_FILE = Path(
    r"C:\asr\index\description\description.index"
)
DESCRIPTION_METADATA_FILE = Path(
    r"C:\asr\index\description\description_metadata.json"
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """Apply consistent text cleaning."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# LOAD ONE INDEX AND ITS METADATA
# ============================================================

def load_index(index_file, metadata_file, text_field):
    """Load a FAISS index and its corresponding metadata."""
    if not index_file.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_file}"
        )

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata not found: {metadata_file}"
        )

    index = faiss.read_index(str(index_file))

    with metadata_file.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    records = metadata["records"]

    if index.ntotal != len(records):
        raise ValueError(
            f"{index_file}: index has {index.ntotal} vectors, "
            f"but metadata has {len(records)} records."
        )

    if index.ntotal > 0:
        if index.d != int(metadata["vector_dimension"]):
            raise ValueError(
                f"Vector dimension mismatch in {index_file}."
            )

    for record in records:
        if text_field not in record:
            raise ValueError(
                f"Metadata record is missing '{text_field}' "
                f"in {metadata_file}."
            )

    return index, records


# ============================================================
# SEARCH ONE INDEX
# ============================================================

def search_index(
    query_embedding,
    index,
    records,
    text_field,
    top_k=TOP_K,
):
    """
    Search an index using an already-created query embedding.

    FAISS IDs map directly to records in the corresponding metadata.
    """
    if index.ntotal == 0:
        return []

    k = min(top_k, index.ntotal)
    scores, ids = index.search(query_embedding, k)

    results = []

    for score, result_id in zip(scores[0], ids[0]):
        result_id = int(result_id)

        if result_id < 0:
            continue

        record = records[result_id]

        results.append({
            "faiss_id": result_id,
            "matched_text": record[text_field],
            "new_intent": record["new_intent"],
            "ivr_intent": record.get("ivr_intent", ""),
            "similarity_score": float(score),
        })

    return results


# ============================================================
# SEARCH BOTH INDICES
# ============================================================

def search_both(query, utterance_index, utterance_records,
                description_index, description_records, model):
    """Search the utterance and description indexes for one query."""
    cleaned_query = clean_text(query)

    if not cleaned_query:
        raise ValueError("Query must not be empty.")

    # Embed the query once; both indexes use the same model.
    embedding = model.encode(
        [cleaned_query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embedding = np.asarray(embedding, dtype=np.float32)

    utterance_results = search_index(
        query_embedding=embedding,
        index=utterance_index,
        records=utterance_records,
        text_field="utterance",
    )

    description_results = search_index(
        query_embedding=embedding,
        index=description_index,
        records=description_records,
        text_field="description",
    )

    return utterance_results, description_results


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(title, results):
    print(f"\n{'=' * 90}")
    print(title)
    print("=" * 90)

    if not results:
        print("No matching records found.")
        return

    for rank, result in enumerate(results, start=1):
        print(f"\nRank:              {rank}")
        print(f"FAISS ID:          {result['faiss_id']}")
        print(f"Matched text:      {result['matched_text']}")
        print(f"New intent:        {result['new_intent']}")
        print(f"IVR intent:        {result['ivr_intent']}")
        print(
            f"Similarity score:  "
            f"{result['similarity_score']:.4f}"
        )
        print("-" * 90)


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading FAISS indexes...")

    utterance_index, utterance_records = load_index(
        UTTERANCE_INDEX_FILE,
        UTTERANCE_METADATA_FILE,
        "utterance",
    )

    description_index, description_records = load_index(
        DESCRIPTION_INDEX_FILE,
        DESCRIPTION_METADATA_FILE,
        "description",
    )

    # Check both indexes were built with the expected model.
    for metadata_file in (
        UTTERANCE_METADATA_FILE,
        DESCRIPTION_METADATA_FILE,
    ):
        with metadata_file.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        if metadata.get("model_name") != MODEL_NAME:
            raise ValueError(
                f"Unexpected embedding model in {metadata_file}: "
                f"{metadata.get('model_name')}"
            )

    # Load the embedding model once.
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Utterance vectors:   {utterance_index.ntotal}")
    print(f"Description vectors: {description_index.ntotal}")
    print(f"Top matches per index: {TOP_K}")
    print("\nEnter a query, or type 'exit' to quit.\n")

    while True:
        query = input("Query: ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        try:
            utterance_results, description_results = search_both(
                query=query,
                utterance_index=utterance_index,
                utterance_records=utterance_records,
                description_index=description_index,
                description_records=description_records,
                model=model,
            )

            print(f"\nInput query: {query}")

            display_results(
                "UTTERANCE INDEX — TOP MATCHES",
                utterance_results,
            )

            display_results(
                "DESCRIPTION INDEX — TOP MATCHES",
                description_results,
            )

            print()

        except Exception as exc:
            print(f"Search failed: {exc}\n")


if __name__ == "__main__":
    main()
