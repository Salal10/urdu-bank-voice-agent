"""The voice loop: browser mic -> Whisper (STT) -> Gemini agent -> MMS (TTS) -> browser speaker, timing each stage.

Run: uv run uvicorn urdu_voice.server:app --port 8000   then open http://127.0.0.1:8000
"""
import base64
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel, decode_audio

from urdu_voice.agent import GREETING, Agent
from urdu_voice.bank import Bank
from urdu_voice.speech_text import speakable
from urdu_voice.stt_eval import SAMPLE_RATE, to_wav_bytes
from urdu_voice.tts import UrduVoice

STATIC = Path(__file__).parent / "static"
DIDNT_HEAR = "معاف کیجیے، آواز صاف نہیں آئی۔ دوبارہ بول دیں۔"

# Loaded once at startup, like singletons: loading takes seconds, using them is what we time.
print("Loading Whisper and the Urdu voice...")
whisper = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
voice = UrduVoice()
bank = Bank()
agent: Agent | None = None  # one call at a time: this is a single-user demo

app = FastAPI()


def listen(audio_file) -> str:
    samples = decode_audio(audio_file, sampling_rate=SAMPLE_RATE)  # browser webm/opus -> 16 kHz floats
    segments, _ = whisper.transcribe(
        samples, language="ur",
        beam_size=1,                       # our bake-off choice: same accuracy, faster
        temperature=0,                     # no fallback retries: they caused hallucinations and 75 s decodes
        condition_on_previous_text=False,
    )
    return " ".join(s.text for s in segments).strip()


def speak(text: str) -> str:
    """Reply text -> spoken wav, base64 so it fits in the JSON response."""
    return base64.b64encode(to_wav_bytes(voice.say(speakable(text)))).decode()


GREETING_AUDIO = speak(GREETING)  # the greeting never changes, so make its audio once at startup


class Timer:
    """Collects how long each stage took, in seconds."""
    def __init__(self):
        self.stages: dict[str, float] = {}
        self._last = time.perf_counter()

    def lap(self, stage: str) -> None:
        now = time.perf_counter()
        self.stages[stage] = round(now - self._last, 2)
        self._last = now


@app.get("/")
def page():
    return FileResponse(STATIC / "index.html")


@app.post("/call")
def new_call():
    """Start a fresh call: the bank goes back to its made-up data and the agent forgets everything."""
    global agent
    bank.reset()
    agent = Agent(bank.new_call())
    return {"reply": GREETING, "audio": GREETING_AUDIO, "timings": {}}


# A plain `def` (not `async def`): FastAPI runs it on a worker thread, so the slow CPU work doesn't block the server.
@app.post("/turn")
def turn(audio: UploadFile):
    if agent is None:
        raise HTTPException(400, "Start a call first.")
    timer = Timer()
    heard = listen(audio.file)
    timer.lap("stt")

    tools_before = len(agent.trace)
    reply = agent.reply(heard) if heard else DIDNT_HEAR
    timer.lap("agent")

    spoken = speak(reply)
    timer.lap("tts")

    timings = timer.stages | {"total": round(sum(timer.stages.values()), 2)}
    print(f"heard: {heard} | reply: {reply} | {timings}")
    return {
        "heard": heard,
        "reply": reply,
        "tools": agent.trace[tools_before:],
        "audio": spoken,
        "timings": timings,
    }