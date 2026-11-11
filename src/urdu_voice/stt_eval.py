"""Speech-to-text bake-off: transcribe the Urdu test clips with one model and score accuracy (CER) and speed."""
import csv
import io
import os
import re
import sys
import time
import wave
from pathlib import Path

import jiwer
import numpy as np
from dotenv import load_dotenv
from faster_whisper import WhisperModel, decode_audio

load_dotenv()

EVAL_DIR = Path("data/eval")
RESULTS_DIR = Path("results")
TEST_SETS = {
    "fleurs": (EVAL_DIR / "fleurs_references.csv", EVAL_DIR / "fleurs"),
    "mine": (EVAL_DIR / "my_references.csv", EVAL_DIR / "mine"),
}
WHISPER_SIZES = ["small", "medium", "large-v3-turbo"]
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
SAMPLE_RATE = 16000

# ---------- Text normalization: only count real recognition mistakes ----------

DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u200C\u200D]")  # zer/zabar/pesh etc. and zero-width joiners
SAME_LETTER = str.maketrans({
    "ي": "ی", "ى": "ی", "ك": "ک", "ه": "ہ", "ۂ": "ہ", "ة": "ہ",  # Arabic forms -> Urdu forms
    **{d: str(i) for i, d in enumerate("۰۱۲۳۴۵۶۷۸۹")},          # Urdu digits -> 0-9
    **{d: str(i) for i, d in enumerate("٠١٢٣٤٥٦٧٨٩")},          # Arabic digits -> 0-9
})
DIGIT_WORDS = {"صفر": "0", "زیرو": "0", "ایک": "1", "دو": "2", "تین": "3", "چار": "4",
               "پانچ": "5", "چھ": "6", "چھے": "6", "سات": "7", "آٹھ": "8", "نو": "9"}


def join_spoken_digits(words: list[str]) -> list[str]:
    """'چار پانچ دو ایک' or '4 5 2 1' -> '4521'. Only runs of 2+, so a lone 'پانچ' stays a word."""
    result, run = [], []
    for word in words + [""]:
        digit = word if word.isdigit() else DIGIT_WORDS.get(word)
        if digit is not None:
            run.append((word, digit))
            continue
        if len(run) >= 2:
            result.append("".join(d for _, d in run))
        else:
            result.extend(w for w, _ in run)
        run = []
        if word:
            result.append(word)
    return result


def normalize(text: str) -> str:
    text = DIACRITICS.sub("", text.translate(SAME_LETTER)).lower()
    text = re.sub(r"[^\w\s]", " ", text)  # punctuation: ۔ ، ؟ . , ? etc.
    return " ".join(join_spoken_digits(text.split()))


def numbers_correct(reference: str, hypothesis: str) -> bool | None:
    """Did every number in the reference (card, account, amount) come out exactly right? None = no numbers."""
    wanted = re.findall(r"\d+", reference)
    if not wanted:
        return None
    heard = re.findall(r"\d+", hypothesis)
    return all(n in heard for n in wanted)


# ---------- Models: each one is a function audio -> text ----------

def whisper_transcriber(size: str):
    model = WhisperModel(size, device="cpu", compute_type="int8")

    def transcribe(audio: np.ndarray) -> str:
        segments, _ = model.transcribe(audio, language="ur", beam_size=5)
        return " ".join(segment.text for segment in segments)
    return transcribe


def gemini_transcriber():
    from google import genai
    from google.genai import types

    client = genai.Client()
    prompt = "Transcribe this audio exactly as spoken, in Urdu script. Output only the transcription."

    def transcribe(audio: np.ndarray) -> str:
        part = types.Part.from_bytes(data=to_wav_bytes(audio), mime_type="audio/wav")
        for attempt in range(5):
            try:
                response = client.models.generate_content(model=GEMINI_MODEL, contents=[prompt, part])
                return response.text or ""
            except Exception as e:  # rate limit / overloaded: wait and retry
                if attempt == 4 or not any(code in str(e) for code in ("429", "503")):
                    raise
                time.sleep(10 * 2 ** attempt)
    return transcribe


def to_wav_bytes(audio: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    return buffer.getvalue()


# ---------- Evaluation ----------

def load_clips() -> list[tuple[str, Path, str]]:
    clips = []
    for set_name, (references, folder) in TEST_SETS.items():
        if not references.exists():
            print(f"Skipping '{set_name}': {references} not found")
            continue
        with references.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                clips.append((set_name, folder / row["clip"], row["reference"]))
    return clips


def main(model_name: str) -> None:
    transcribe = gemini_transcriber() if model_name == "gemini" else whisper_transcriber(model_name)
    rows = []
    for set_name, path, reference in load_clips():
        audio = decode_audio(str(path), sampling_rate=SAMPLE_RATE)  # wav or m4a -> 16 kHz mono floats
        start = time.perf_counter()
        hypothesis = transcribe(audio)
        seconds = time.perf_counter() - start

        ref, hyp = normalize(reference), normalize(hypothesis)
        rows.append({
            "set": set_name,
            "clip": path.name,
            "reference": reference,
            "hypothesis": hypothesis.strip(),
            # Spaces removed: Urdu word spacing is inconsistent ("چیک بک" vs "چیکبک") and isn't a real mistake.
            "cer": round(jiwer.cer(ref.replace(" ", ""), hyp.replace(" ", "")), 4),
            "numbers_ok": numbers_correct(ref, hyp),
            "audio_seconds": round(len(audio) / SAMPLE_RATE, 2),
            "seconds": round(seconds, 2),
        })
        print(f"{path.name}: CER {rows[-1]['cer']:.2f}  {seconds:.1f}s  {hypothesis.strip()[:60]}")

    RESULTS_DIR.mkdir(exist_ok=True)
    output = RESULTS_DIR / f"stt_{model_name}.csv"
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{model_name}  (saved to {output})")
    for set_name in TEST_SETS:
        part = [r for r in rows if r["set"] == set_name]
        if not part:
            continue
        cer = sum(r["cer"] for r in part) / len(part)
        rtf = sum(r["seconds"] for r in part) / sum(r["audio_seconds"] for r in part)
        checked = [r["numbers_ok"] for r in part if r["numbers_ok"] is not None]
        numbers = f"{sum(checked)}/{len(checked)}" if checked else "-"
        print(f"  {set_name:7} clips {len(part):3}  CER {cer:.1%}  real-time factor {rtf:.2f}  numbers right {numbers}")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "small"
    if name not in WHISPER_SIZES + ["gemini"]:
        sys.exit(f"Model must be one of: {', '.join(WHISPER_SIZES)}, gemini")
    main(name)