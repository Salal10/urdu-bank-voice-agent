# Urdu bank voice agent

A phone-style assistant for bank customers who speak Urdu. You hold a button, say "my card is lost, block it" in Urdu, and it checks who you are, reads the card number back, blocks the card and answers out loud in Urdu.

Everything uses free models:

| Stage | Model | Where it runs |
|---|---|---|
| Speech to text | Whisper large-v3-turbo (faster-whisper, int8) | locally, CPU |
| Understanding + actions | Gemini Flash-Lite with tool calling | Google API (text only) |
| Text to speech | Meta MMS-TTS Urdu | locally, CPU |

The audio never leaves the machine. Only the transcribed text goes to Gemini. All customers, accounts and numbers are made up.

## How a turn works

```
browser mic -> Whisper -> digits fixed in code -> Gemini + bank tools -> numbers spelled out in Urdu -> MMS -> browser speaker
```

The bank (`bank.py`) is a small Python module with three made-up customers. The rules that matter are in code, not in the prompt:

- No balance, transactions, cards, blocking or complaints until the caller gives a matching account number and the last 4 digits of their CNIC.
- 3 wrong attempts lock the call.
- A caller can only block their own cards.
- Blocking takes two steps. The confirm step refuses unless the caller has spoken since the card number was read back, so the model can't skip the confirmation.
- Talking to a person always works, verified or not.

## Choosing the speech-to-text model

I tested 40 clips from FLEURS (read Urdu speech) and 20 banking phrases I recorded myself, with card numbers, amounts and English words mixed in. Error rate is character error rate (CER), after normalising spelling variants and digits.

| Model | FLEURS CER | My banking phrases CER | Seconds per sentence (CPU) |
|---|---|---|---|
| Whisper small | 16.2% | 17.7% | ~1.6 |
| Whisper medium | 11.5% | 10.9% | ~7 |
| **Whisper large-v3-turbo** | **10.2%** | **7.8%** | ~6-10 |
| Gemini (audio in) | 15.4% | 53.8% | - |

Gemini was very accurate when it answered in Urdu script (under 5% CER), but in 9 of 20 banking phrases it answered in Roman Urdu or Hindi script, so it can't be used as is. Giving Whisper a list of banking words as a hint didn't help turbo (8.4% vs 7.8%).

## Making the voice say numbers

MMS reads "25000" badly, so replies go through `speakable()` first. Amounts become Urdu words (پچیس ہزار روپے), and long numbers like account numbers are read digit by digit. I tested it by letting Whisper listen to the generated speech:

| | CER heard back | Replies with every number heard right |
|---|---|---|
| Reply as written | 18.6% | 0/8 |
| After `speakable()` | 13.2% | 5/8 |

## End-to-end test

`e2e_eval.py` plays 9 scripted phone calls (32 turns) through the whole loop, using my recordings as the caller. A few lines the recordings don't have, like the CNIC digits, are spoken by the MMS voice so they still go through Whisper. Each call is graded on what the bank actually did (its action log), not on what the agent said.

| Version | Calls with the right outcome | Replies in Urdu script |
|---|---|---|
| v1 | 6/9 | 29/29 |
| v2 | 9/9 | 32/32 |

What v1 got wrong:

- It said "I'm transferring you to an officer" and "I'll file a complaint" without ever calling the tool. Grading the replies would have passed both.
- It told the caller branches open Monday to Thursday when the tool said Monday to Friday.

What changed in v2:

- After each reply, a check in code looks for phrases that claim an action happened. If the matching tool didn't succeed in that turn, the model gets one automatic correction: call the tool, or take the claim back.
- Tool results are written ready to say in Urdu, so there's nothing left to translate.
- For disputed transactions, the agent looks up the transaction and uses the bank's amount, not the one it thinks it heard.

Testing with real voice also found problems that typed tests didn't:

- Whisper writes صفر (zero) as سفر, which turned 10473382 into 17473382. Spoken digits are now converted to numbers in code before the model sees them.
- An empty reply from the model crashed the voice. Both now have guards.

## Speed

Median seconds per turn on a laptop CPU, no GPU (v1 run on a quiet machine):

| Speech to text | Agent | Text to speech | Total |
|---|---|---|---|
| 6.8 | 1.6 | 2.0 | 10.4 |

Speech recognition is most of the wait. Running Whisper alone was just as slow, and more CPU threads didn't help. A real phone call needs replies in 1-2 seconds, so this needs a GPU or streaming recognition. On a busy machine the same test took over twice as long.

## Limitations

- Too slow for real calls on CPU (see above).
- MMS pronunciation is weak. Its alphabet has no short-vowel marks, so it guesses vowels, especially on English words written in Urdu. It also drops ۔ ، ؟, so sentences run together.
- MMS is licensed CC-BY-NC (non-commercial).
- All recordings are my own voice. The CNIC lines in the test are synthetic. My recordings aren't in the repo.
- One run per test call. Gemini's choices vary between runs even at temperature 0, so 9/9 is one run, not a guarantee.
- One call at a time. A real server would keep an agent per session.

## Run it

Needs Python 3.13 and [uv](https://docs.astral.sh/uv/). Put a Gemini API key in `.env`:

```
GEMINI_API_KEY=your-key
```

```
uv sync
uv run python -m urdu_voice.bank          # bank self-check, no API key needed
uv run python -m urdu_voice.agent         # scripted typed call through the agent
uv run uvicorn urdu_voice.server:app --port 8000
```

Then open http://127.0.0.1:8000, press "New call", and hold the button (or Space) while you speak. The first start downloads the Whisper and MMS models.

The speech tests (`stt_eval`, `tts_eval`, `e2e_eval`) need the FLEURS clips (`uv run python -m urdu_voice.prepare_fleurs`) and your own recordings in `data/eval/mine/`, matching `data/eval/my_references.csv`.
