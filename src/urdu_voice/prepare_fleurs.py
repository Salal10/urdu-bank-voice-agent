import csv
import random
from pathlib import Path

import httpx
import pyarrow.parquet as pq

URL = "https://huggingface.co/api/datasets/google/fleurs/parquet/ur_pk/test/0.parquet"
RAW_PATH = Path("data/raw/fleurs_ur_test.parquet")
CLIPS_DIR = Path("data/eval/fleurs")
REFERENCES_PATH = Path("data/eval/fleurs_references.csv")
SAMPLE_SIZE = 40
SEED = 42


def download() -> None:
    if RAW_PATH.exists():
        print(f"Already downloaded: {RAW_PATH}")
        return
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading FLEURS Urdu test set (about 190 MB)...")
    with httpx.stream("GET", URL, follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        with RAW_PATH.open("wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)


def sample_clips() -> None:
    table = pq.read_table(RAW_PATH, columns=["id", "audio", "raw_transcription"])
    rows = table.to_pylist()

    # FLEURS has several recordings of the same sentence: keep one per sentence.
    unique = list({row["id"]: row for row in rows}.values())
    sample = random.Random(SEED).sample(unique, SAMPLE_SIZE)

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    with REFERENCES_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["clip", "reference"])
        for row in sample:
            name = f"fleurs_{row['id']}.wav"
            (CLIPS_DIR / name).write_bytes(row["audio"]["bytes"])
            writer.writerow([name, row["raw_transcription"]])
    print(f"Saved {len(sample)} clips to {CLIPS_DIR} and references to {REFERENCES_PATH}")


if __name__ == "__main__":
    download()
    sample_clips()