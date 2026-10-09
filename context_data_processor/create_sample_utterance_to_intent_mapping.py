
import re
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(r"C:\asr\intents.csv")
OUTPUT_FILE = Path(r"C:\utterance_intent_mapping.csv")

NEW_INTENT_COLUMN = "New Taxonomy Intents"
IVR_INTENT_COLUMN = "Mapping to IVR intents"
UTTERANCE_COLUMN = "Sample Utterances"


# ============================================================
# CLEANING
# ============================================================

def clean_field(value):
    """
    Keep letters, digits, and spaces.
    Normalize whitespace to a single space.
    Remove leading and trailing whitespace.
    Remove quotation marks, underscores, and punctuation.
    """
    if pd.isna(value):
        return ""

    value = str(value)

    # Replace consecutive whitespace with one space.
    value = re.sub(r"\s+", " ", value).strip()

    # Keep only ASCII letters, digits, and spaces.
    value = re.sub(r"[^a-zA-Z0-9 ]", " ", value)

    return value.strip()


def extract_utterances(text):
    """Extract individual utterances from multiline cells."""
    if not text:
        return []

    utterances = []

    for line in str(text).splitlines():
        utterance = clean_field(line)

        if utterance:
            utterances.append(utterance)

    return utterances


# ============================================================
# MAIN
# ============================================================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_FILE}"
        )

    # Read source CSV.
    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
    )

    # Clean column headers.
    df.columns = df.columns.str.strip()

    # Validate required columns.
    required_columns = [
        NEW_INTENT_COLUMN,
        IVR_INTENT_COLUMN,
        UTTERANCE_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required CSV columns: {missing_columns}"
        )

    # Create utterance-to-intent mappings.
    mappings = []
    seen = set()

    for _, row in df.iterrows():
        new_intent = clean_field(row[NEW_INTENT_COLUMN])
        ivr_intent = clean_field(row[IVR_INTENT_COLUMN])

        sample_text = row[UTTERANCE_COLUMN]

        if not new_intent:
            continue

        for utterance in extract_utterances(sample_text):
            # Case-insensitive duplicate detection.
            key = (
                utterance.casefold(),
                ivr_intent.casefold(),
                new_intent.casefold(),
            )

            if key in seen:
                continue

            seen.add(key)

            mappings.append({
                "utterance": utterance,
                "ivr_intent": ivr_intent,
                "new_intent": new_intent,
            })

    # Build output DataFrame.
    output_df = pd.DataFrame(
        mappings,
        columns=[
            "utterance",
            "ivr_intent",
            "new_intent",
        ],
    )

    # Clean every output field.
    for column in output_df.columns:
        output_df[column] = output_df[column].map(clean_field)

    # Save output CSV.
    output_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # Report results.
    print(f"Input file: {INPUT_FILE}")
    print(f"Input rows: {len(df)}")
    print(f"Output mappings: {len(output_df)}")
    print(f"Output file: {OUTPUT_FILE}")
    print(output_df.head(100))

if __name__ == "__main__":
    main()
