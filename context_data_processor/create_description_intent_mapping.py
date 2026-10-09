
import re
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(r"C:\asr\intents.csv")
OUTPUT_FILE = Path(r"C:\asr\description_intent_mapping.csv")

NEW_INTENT_COLUMN = "New Taxonomy Intents"
IVR_INTENT_COLUMN = "Mapping to IVR intents"
DESCRIPTION_COLUMN = "Description"


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

    # Normalize whitespace.
    value = re.sub(r"\s+", " ", value).strip()

    # Keep only ASCII letters, digits, and spaces.
    value = re.sub(r"[^a-zA-Z0-9 ]", " ", value)

    # Collapse spaces introduced by punctuation removal.
    value = re.sub(r"\s+", " ", value).strip()

    return value


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
        DESCRIPTION_COLUMN,
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

    # Create description-to-intent mappings.
    mappings = []
    seen = set()

    for _, row in df.iterrows():
        new_intent = clean_field(row[NEW_INTENT_COLUMN])
        ivr_intent = clean_field(row[IVR_INTENT_COLUMN])
        description = clean_field(row[DESCRIPTION_COLUMN])

        # Skip records without a description or new intent.
        if not description or not new_intent:
            continue

        # Case-insensitive duplicate detection.
        key = (
            description.casefold(),
            ivr_intent.casefold(),
            new_intent.casefold(),
        )

        if key in seen:
            continue

        seen.add(key)

        mappings.append({
            "description": description,
            "ivr_intent": ivr_intent,
            "new_intent": new_intent,
        })

    # Build output DataFrame.
    output_df = pd.DataFrame(
        mappings,
        columns=[
            "description",
            "ivr_intent",
            "new_intent",
        ],
    )

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
    print("\nFirst 100 mappings:")
    print(output_df.head(100).to_string(index=False))


if __name__ == "__main__":
    main()
