"""
Preprocess Vietnamese datasets for pseudo-labeling.

Extracts text from 5 Vietnamese NLP datasets (train splits only),
cleans them (remove emojis, special chars, URLs, etc.), deduplicates,
and saves to a unified CSV for LLM pseudo-labeling.

Usage:
    python preprocess_datasets.py
"""

import os
import re
import csv
import unicodedata
from collections import OrderedDict


# ── Dataset configs ──────────────────────────────────────────────────
DATASETS = {
    "UIT-VSMEC": {
        "path": "data/UIT-VSMEC/train.csv",
        "text_col": "Sentence",
    },
    "UIT-ViSFD": {
        "path": "data/UIT-ViSFD/Train.csv",
        "text_col": "comment",
    },
    "ViHOS": {
        "path": "data/ViHOS/train.csv",
        "text_col": "content",
    },
    "ViHSD": {
        "path": "data/ViHSD/train.csv",
        "text_col": "free_text",
    },
    "ViSpamReview": {
        "path": "data/ViSpamReview/train.csv",
        "text_col": "Comment",
    },
}

OUTPUT_DIR = "data/pseudo_label"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "unlabeled_texts.csv")

# Min/max text length (in chars) after cleaning
MIN_TEXT_LEN = 5
MAX_TEXT_LEN = 512


# ── Emoji & special character patterns ───────────────────────────────
# Matches most emoji ranges
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U0001F251"  # enclosed chars
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols extended-A
    "\U00002600-\U000026FF"  # misc symbols
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"             # ZWJ
    "\U00020000-\U0002FA1F"  # CJK
    "]+",
    flags=re.UNICODE,
)

URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+", flags=re.IGNORECASE
)

MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")

# Multiple whitespace / newlines
MULTI_SPACE = re.compile(r"\s+")

# Non-Vietnamese / non-printable junk (keep ONLY letters and whitespace)
ALLOWED_CHARS = re.compile(
    r"[^"
    r"a-zA-Z"              # ASCII letters
    r"\u00C0-\u024F"       # Latin Extended (covers most Vietnamese)
    r"\u0300-\u036F"       # Combining diacritical marks
    r"\u1EA0-\u1EF9"       # Vietnamese-specific block
    r"\s"                  # whitespace
    r"]"
)


def clean_text(text: str) -> str:
    """Clean a single text string for pseudo-labeling."""
    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = URL_PATTERN.sub("", text)

    # Remove mentions and hashtags
    text = MENTION_PATTERN.sub("", text)
    text = HASHTAG_PATTERN.sub("", text)

    # Remove emojis
    text = EMOJI_PATTERN.sub("", text)

    # Normalize unicode (NFC form for Vietnamese)
    text = unicodedata.normalize("NFC", text)

    # Remove remaining special / non-Vietnamese chars
    text = ALLOWED_CHARS.sub(" ", text)

    # Collapse whitespace
    text = MULTI_SPACE.sub(" ", text).strip()

    return text


def is_valid_text(text: str) -> bool:
    """Check if cleaned text is valid for pseudo-labeling."""
    if len(text) < MIN_TEXT_LEN:
        return False
    if len(text) > MAX_TEXT_LEN:
        return False

    # Must contain at least some Vietnamese/Latin letters
    letter_count = sum(1 for c in text if c.isalpha())
    if letter_count < 3:
        return False

    # Skip if mostly non-Vietnamese (heuristic: >50% non-alpha non-space)
    non_alpha = sum(1 for c in text if not c.isalpha() and not c.isspace())
    if len(text) > 10 and non_alpha / len(text) > 0.7:
        return False

    return True


def load_dataset(name: str, config: dict) -> list[dict]:
    """Load a single dataset and return list of {text, source}."""
    path = config["path"]
    text_col = config["text_col"]

    if not os.path.exists(path):
        print(f"  ⚠ File not found: {path}, skipping {name}")
        return []

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_text = row.get(text_col, "").strip()
            cleaned = clean_text(raw_text)
            if is_valid_text(cleaned):
                rows.append({"text": cleaned, "source": name})

    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_texts = []
    stats = {}

    print("=" * 60)
    print("  Preprocessing datasets for pseudo-labeling")
    print("=" * 60)

    for name, config in DATASETS.items():
        print(f"\n📂 Loading {name}...")
        rows = load_dataset(name, config)
        stats[name] = {"loaded": len(rows)}
        all_texts.extend(rows)
        print(f"   ✓ {len(rows)} valid texts after cleaning")

    # Deduplicate by text content (keep first occurrence)
    print(f"\n🔄 Deduplicating {len(all_texts)} total texts...")
    seen = OrderedDict()
    for item in all_texts:
        if item["text"] not in seen:
            seen[item["text"]] = item

    deduped = list(seen.values())
    removed = len(all_texts) - len(deduped)
    print(f"   ✓ Removed {removed} duplicates, {len(deduped)} unique texts remain")

    # Write output
    print(f"\n💾 Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "source"])
        writer.writeheader()
        writer.writerows(deduped)

    # Print summary stats
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    source_counts = {}
    for item in deduped:
        source_counts[item["source"]] = source_counts.get(item["source"], 0) + 1

    for name in DATASETS:
        loaded = stats[name]["loaded"]
        final = source_counts.get(name, 0)
        print(f"  {name:15s}: {loaded:>6} loaded → {final:>6} after dedup")

    print(f"  {'TOTAL':15s}: {len(all_texts):>6} loaded → {len(deduped):>6} after dedup")
    print(f"\n  Output: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
