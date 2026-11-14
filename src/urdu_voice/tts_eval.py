"""Text-to-speech test: speak typical bank replies, then let Whisper listen back and check what it heard."""
import csv
import time
from pathlib import Path

import jiwer

from urdu_voice.speech_text import speakable, spoken_numbers, unspeakable_words
from urdu_voice.stt_eval import normalize, to_wav_bytes, whisper_transcriber
from urdu_voice.tts import UrduVoice

AUDIO_DIR = Path("data/tts")
RESULTS_PATH = Path("results/tts_mms.csv")

# Replies the agent will give (made-up data), including numbers and one English word in Latin script.
REPLIES = [
    "آپ کے اکاؤنٹ کا بیلنس 25,000 روپے ہے۔",
    "آپ کے کارڈ کے آخری چار نمبر 4521 ہیں، کیا یہ درست ہے؟",
    "آپ کا کارڈ بلاک کر دیا گیا ہے۔",
    "آپ کا اکاؤنٹ نمبر 10473382 ہے، کیا یہ درست ہے؟",
    "آپ کی آخری ٹرانزیکشن 3,500 روپے کی تھی۔",
    "آپ کی قسط 12,750 روپے ہے جو مہینے کی پہلی تاریخ کو کٹے گی۔",
    "آپ کی شکایت درج کر لی گئی ہے، آپ کا شکایت نمبر 58213 ہے۔",
    "برانچ صبح 9 بجے کھلتی ہے۔",
    "میں آپ کو ہمارے نمائندے سے ملا رہا ہوں، براہ کرم انتظار کریں۔",
    "نئی چیک بک سات دن میں آپ کے پتے پر پہنچ جائے گی۔",
    "آپ کی تنخواہ 1,25,000 روپے آپ کے اکاؤنٹ میں آ گئی ہے۔",
    "آپ کا کارڈ block کر دیا گیا ہے۔",
]


def compact(text: str) -> str:
    """Same form for both sides: numbers as Urdu words, normalized, no spaces. Whisper may write
    "25000" or "پچیس ہزار" for the same sound, and both should count as right."""
    return normalize(speakable(text)).replace(" ", "")


def numbers_heard(reply: str, heard: str) -> bool | None:
    """Did every number in the reply come through clearly enough for a listener to get it right?"""
    expected = spoken_numbers(reply)
    if not expected:
        return None
    return all(compact(number) in compact(heard) for number in expected)


def main() -> None:
    voice = UrduVoice()
    listen = whisper_transcriber("large-v3-turbo")  # the "listener" that checks if the speech is understandable
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, reply in enumerate(REPLIES, start=1):
        # "as_written": the reply as the agent wrote it. "speakable": numbers turned into Urdu words first.
        for version, text in [("as_written", reply), ("speakable", speakable(reply))]:
            start = time.perf_counter()
            audio = voice.say(text)
            seconds = time.perf_counter() - start
            (AUDIO_DIR / f"{i:02}_{version}.wav").write_bytes(to_wav_bytes(audio))

            heard = listen(audio)
            rows.append({
                "reply": i,
                "version": version,
                "text": text,
                "heard": heard.strip(),
                "cer": round(jiwer.cer(compact(reply), compact(heard)), 4),
                "numbers_ok": numbers_heard(reply, heard),
                "skipped_words": " ".join(unspeakable_words(text)),
                "audio_seconds": round(len(audio) / voice.sample_rate, 2),
                "seconds": round(seconds, 2),
            })
            print(f"{i:02} {version:10} CER {rows[-1]['cer']:.2f}  {seconds:.2f}s  heard: {heard.strip()}")

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved audio to {AUDIO_DIR} and results to {RESULTS_PATH}")
    for version in ("as_written", "speakable"):
        part = [r for r in rows if r["version"] == version]
        cer = sum(r["cer"] for r in part) / len(part)
        checked = [r["numbers_ok"] for r in part if r["numbers_ok"] is not None]
        speed = sum(r["seconds"] for r in part) / len(part)
        print(f"  {version:10}  CER heard back {cer:.1%}  numbers right {sum(checked)}/{len(checked)}  "
              f"seconds per reply {speed:.2f}")


if __name__ == "__main__":
    main()