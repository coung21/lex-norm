"""
Pseudo-label Vietnamese texts using Gemini 2.5 Flash Lite via OpenAI-compatible API.

Sends batches of texts to the LLM for lexical normalization, with:
- System prompt for normalization rules
- 20 texts per request (JSON array)
- Async concurrent requests with rate limiting
- Checkpoint/resume support

Usage:
    # Set API key
    export GOOGLE_API_KEY="your-api-key"

    # Dry run (first 10 texts)
    python pseudo_label.py --limit 10

    # Full run
    python pseudo_label.py

    # Resume from checkpoint
    python pseudo_label.py --resume
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path
from collections import OrderedDict

from openai import AsyncOpenAI

# ── Config ───────────────────────────────────────────────────────────
BATCH_SIZE = 50          # Giảm xuống 50 để JSON ổn định hơn, không bị lỗi Unterminated string
MAX_CONCURRENT = 3       # Giữ 3 luồng để đảm bảo throughput ổn định
CHECKPOINT_INTERVAL = 5  
MAX_RETRIES = 5         
BASE_DELAY = 5          
FIXED_DELAY = 2          # Nghỉ 2s sau mỗi batch lớn

INPUT_FILE = "data/pseudo_label/unlabeled_texts.csv"
OUTPUT_FILE = "data/pseudo_label/pseudo_labeled.csv"
CHECKPOINT_FILE = "data/pseudo_label/.checkpoint.json"

MODEL = "deepseek-chat" 

SYSTEM_PROMPT = """Bạn là chuyên gia chuẩn hóa văn bản tiếng Việt trên mạng xã hội.

Nhiệm vụ của bạn là chuyển các câu tiếng Việt không chuẩn (teencode, viết tắt, tiếng lóng, phương ngữ, sai chính tả) thành câu tiếng Việt chuẩn (canonical form).
Canonical form là tiếng Việt chuẩn chính tả, dạng văn viết chính thức.


Quy tắc:
- Chỉ sửa lỗi chính tả và viết tắt/teencode,v.v.. KHÔNG thay đổi ý nghĩa câu
- Giữ nguyên dấu câu và cấu trúc gốc
- Không thêm ký tự
- Giữ nguyên từ đã đúng, chỉ chuẩn hoá từ không chuẩn
- Không thêm nội dung mới
- Trả về JSON array với đúng thứ tự tương ứng, KHÔNG giải thích gì thêm
"""


def load_texts(path: str) -> list[dict]:
    """Load texts from preprocessed CSV."""
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append({"text": row["text"], "source": row["source"]})
    return texts


def load_processed_texts(path: str) -> set[str]:
    """Load already processed original texts from CSV to avoid re-processing."""
    if not os.path.exists(path):
        return set()
    
    processed = set()
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed.add(row["original"])
    return processed


def append_results(results: list[dict], path: str):
    """Append new results to the CSV file immediately."""
    file_exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["original", "normalized", "source"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)


def save_results(results: list[dict], path: str):
    """Save results to CSV."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["original", "normalized", "source"])
        writer.writeheader()
        writer.writerows(results)


def create_batches(texts: list[dict], batch_size: int) -> list[list[dict]]:
    """Split texts into batches."""
    return [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]


async def process_batch(
    client: AsyncOpenAI,
    batch: list[dict],
    semaphore: asyncio.Semaphore,
    batch_idx: int,
) -> list[dict]:
    """Process a single batch of texts through the LLM."""
    texts = [item["text"] for item in batch]
    sources = [item["source"] for item in batch]

    user_message = json.dumps(texts, ensure_ascii=False)

    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=0.1,
                    max_tokens=8192, # Tăng max_tokens để chứa được 100 câu output
                )

            content = response.choices[0].message.content.strip()

            # Parse JSON response - handle markdown code blocks
            if content.startswith("```"):
                # Remove ```json ... ``` wrapper
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

            normalized = json.loads(content)

            if not isinstance(normalized, list):
                raise ValueError(f"Expected list, got {type(normalized)}")

            if len(normalized) != len(texts):
                print(f"  ⚠ Batch {batch_idx}: expected {len(texts)} results, "
                      f"got {len(normalized)}, retrying...")
                raise ValueError("Length mismatch")

            # Build results
            results = []
            for orig, norm, src in zip(texts, normalized, sources):
                norm = str(norm).strip()
                if norm:
                    results.append({
                        "original": orig,
                        "normalized": norm,
                        "source": src,
                    })

            return results

        except (json.JSONDecodeError, ValueError) as e:
            delay = BASE_DELAY * (2 ** attempt)
            print(f"  ⚠ Batch {batch_idx} parse error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(delay)
            else:
                print(f"  ✗ Batch {batch_idx} FAILED after {MAX_RETRIES} attempts, skipping")
                return []

        except Exception as e:
            delay = BASE_DELAY * (2 ** attempt)
            error_str = str(e).lower()
            if "rate" in error_str or "429" in error_str or "quota" in error_str:
                print(f"  ⏳ Rate limited (batch {batch_idx}), waiting {delay}s... (Detail: {error_str[:120]})")
            else:
                print(f"  ⚠ Batch {batch_idx} error: {e}")

            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(delay)
            else:
                print(f"  ✗ Batch {batch_idx} FAILED after {MAX_RETRIES} attempts, skipping")
                return []

    return []


async def run_pseudo_labeling(
    texts: list[dict],
):
    """Main async loop for pseudo-labeling."""
    # 1. Check what's already done
    processed = load_processed_texts(OUTPUT_FILE)
    to_process = [t for t in texts if t["text"] not in processed]
    
    if len(processed) > 0:
        print(f"  ▶ Resuming: {len(processed)} already done, {len(to_process)} remaining.")
    
    if not to_process:
        print("  ✓ All texts already processed!")
        return []

    # 2. Setup API
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: Set DEEPSEEK_API_KEY environment variable")
        sys.exit(1)

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    # 3. Prepare batches
    batches = create_batches(to_process, BATCH_SIZE)
    total_batches = len(batches)

    print(f"\n  Total remaining: {len(to_process)}")
    print(f"  Batch size:      {BATCH_SIZE}")
    print(f"  Total batches:   {total_batches}")
    print(f"  Model:           {MODEL}")

    # 4. Process batches 
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    completed = 0
    failed = 0
    start_time = time.time()

    # Process in chunks
    for chunk_start in range(0, total_batches, CHECKPOINT_INTERVAL):
        chunk_end = min(chunk_start + CHECKPOINT_INTERVAL, total_batches)
        chunk_batches = batches[chunk_start:chunk_end]

        tasks = [
            process_batch(client, batch, semaphore, chunk_start + i)
            for i, batch in enumerate(chunk_batches)
        ]

        chunk_results = await asyncio.gather(*tasks)
        
        # Save results of this chunk immediately
        total_new_results = []
        for batch_result in chunk_results:
            if batch_result:
                total_new_results.extend(batch_result)
            else:
                failed += 1
            completed += 1
        
        if total_new_results:
            append_results(total_new_results, OUTPUT_FILE)

        # Progress
        elapsed = time.time() - start_time
        rate = completed / elapsed if elapsed > 0 else 0
        eta = (total_batches - completed) / rate if rate > 0 else 0
        print(f"  📊 Progress: {completed}/{total_batches} batches | "
              f"Added {len(total_new_results)} results | {failed} failed batches | "
              f"ETA: {eta:.0f}s")

    return []


def main():
    parser = argparse.ArgumentParser(
        description="Pseudo-label Vietnamese texts using Gemini"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of texts to process (for testing)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Pseudo-Labeling with Gemini 2.5 Flash Lite")
    print("=" * 60)

    # Load texts
    print(f"\n📂 Loading texts from {INPUT_FILE}...")
    texts = load_texts(INPUT_FILE)
    print(f"   ✓ Loaded {len(texts)} texts")

    if args.limit:
        texts = texts[:args.limit]
        print(f"   ⚡ Limited to {args.limit} texts (test mode)")

    # Run pseudo-labeling
    asyncio.run(run_pseudo_labeling(texts))

    # Stats - Read from final file
    print("\n" + "=" * 60)
    print("  Summary (Final File)")
    print("=" * 60)

    if os.path.exists(OUTPUT_FILE):
        source_counts = {}
        same_count = 0
        total = 0
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                total += 1
                source_counts[r["source"]] = source_counts.get(r["source"], 0) + 1
                if r["original"] == r["normalized"]:
                    same_count += 1

        for source, count in sorted(source_counts.items()):
            print(f"  {source:15s}: {count:>6} labeled")
        print(f"  {'TOTAL':15s}: {total:>6} labeled")
        print(f"  Unchanged:      {same_count:>6} ({same_count/max(total,1)*100:.1f}%)")
    else:
        print("  ✗ No output file found.")
    print("=" * 60)


if __name__ == "__main__":
    main()
