"""End-to-end test: scripted phone calls, spoken, through the full loop (Whisper -> agent -> MMS).

Each call is graded on what the bank actually did (the action log), not on what the agent said.
Run: uv run python -m urdu_voice.e2e_eval
"""
import csv
import io
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from urdu_voice.agent import Agent
from urdu_voice.bank import Bank
from urdu_voice.server import listen, voice  # same Whisper settings and voice as the web page
from urdu_voice.speech_text import speakable
from urdu_voice.stt_eval import to_wav_bytes

RECORDINGS = Path("data/eval/mine")
RESULTS_PATH = Path("results/e2e.csv")

# "my_09.m4a" = one of your recordings. "tts:..." = a line none of the recordings has (like the CNIC digits),
# spoken by the MMS voice so it still goes through Whisper.
VERIFY = ["my_09.m4a", "tts:میرے شناختی کارڈ کے آخری چار ہندسے چار تین دو ایک ہیں"]


@dataclass
class Result:
    done: set[tuple[str, str]]  # bank actions that succeeded: ("block_card", "4521"), ("complaint", "CardIssue")...
    tools: set[str]             # tools that returned data, not an error
    said: str                   # everything the agent said, commas removed so "25,000" matches "25000"


@dataclass
class Case:
    name: str
    turns: list[str]
    passed: Callable[[Result], bool]


def did(result: Result, action: str) -> bool:
    return any(a == action for a, _ in result.done)


CASES = [
    Case("lost card: block it", ["my_01.m4a", *VERIFY, "my_04.m4a", "my_17.m4a"],
         lambda r: ("block_card", "4521") in r.done),
    Case("balance", ["my_02.m4a", *VERIFY],
         lambda r: "get_balance" in r.tools and "25000" in r.said),
    Case("withdrew 15,000, 25,000 deducted", ["my_05.m4a", *VERIFY, "my_17.m4a"],
         lambda r: ("complaint", "DisputedTransaction") in r.done),
    Case("salary not received", ["my_12.m4a", *VERIFY, "my_17.m4a"],
         lambda r: ("complaint", "SalaryNotReceived") in r.done),
    Case("online transfer failed, money deducted", ["my_13.m4a", *VERIFY, "my_17.m4a"],
         lambda r: ("complaint", "DisputedTransaction") in r.done or ("complaint", "Other") in r.done),
    Case("branch hours (no verification needed)", ["my_07.m4a"],
         lambda r: "get_branch_hours" in r.tools and re.search("جمع[ہے]", r.said)),  # must say open until Friday
    Case("asks for a human", ["my_11.m4a"],
         lambda r: did(r, "handoff")),
    Case("caller says the read-back is wrong: don't block", ["my_01.m4a", *VERIFY, "my_04.m4a", "my_18.m4a"],
         lambda r: not did(r, "block_card")),
    Case("3 wrong CNICs: no data, locked, handed off",
         ["my_02.m4a", "my_09.m4a",
          "tts:شناختی کارڈ کے آخری چار ہندسے ایک ایک ایک ایک ہیں",
          "tts:اچھا تو دو دو دو دو",
          "tts:تین تین تین تین"],
         lambda r: "get_balance" not in r.tools and not did(r, "verify") and did(r, "handoff")),
]


def caller_audio(turn: str):
    if turn.startswith("tts:"):
        return io.BytesIO(to_wav_bytes(voice.say(speakable(turn[4:]))))
    return str(RECORDINGS / turn)


def in_urdu_script(text: str) -> bool:
    """No Latin or Devanagari letters, and at least some Arabic-script letters."""
    return bool(re.search(r"[\u0600-\u06FF]", text)) and not re.search(r"[A-Za-z\u0900-\u097F]", text)


def run_call(case: Case) -> tuple[bool, list[dict], int]:
    bank = Bank()
    agent = Agent(bank.new_call())
    rows, replies = [], []
    for number, turn in enumerate(case.turns, start=1):
        audio = caller_audio(turn)
        t0 = time.perf_counter()
        heard = listen(audio)
        t1 = time.perf_counter()
        tools_before = len(agent.trace)
        reply = agent.reply(heard)
        t2 = time.perf_counter()
        voice.say(speakable(reply))
        t3 = time.perf_counter()
        replies.append(reply)
        rows.append({
            "call": case.name, "turn": number, "audio": turn, "heard": heard, "reply": reply,
            "tools": "; ".join(f"{t['tool']}{t['args']}" for t in agent.trace[tools_before:]),
            "urdu_script": in_urdu_script(reply),
            "stt": round(t1 - t0, 2), "agent": round(t2 - t1, 2), "tts": round(t3 - t2, 2),
            "total": round(t3 - t0, 2),
        })
        print(f"  {number}. heard: {heard}\n     reply: {reply}  ({t3 - t0:.1f}s)")

    result = Result(
        done={(a["action"], a["detail"]) for a in bank.actions if a["result"] == "ok"},
        tools={t["tool"] for t in agent.trace if not (isinstance(t["result"], dict) and "error" in t["result"])},
        said=" ".join(replies).replace(",", ""),
    )
    # How often the code stopped the model (a refused tool call): the guardrails doing their job.
    refusals = sum(1 for t in agent.trace if isinstance(t["result"], dict) and "error" in t["result"])
    return bool(case.passed(result)), rows, refusals


def main() -> None:
    all_rows, passed, refusals = [], [], 0
    for case in CASES:
        print(f"\n=== {case.name}")
        ok, rows, refused = run_call(case)
        print(f"  -> {'PASS' if ok else 'FAIL'}")
        passed.append((case.name, ok))
        all_rows += rows
        refusals += refused

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved every turn to {RESULTS_PATH}\n")
    print(f"Calls with the right outcome: {sum(ok for _, ok in passed)}/{len(passed)}")
    for name, ok in passed:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    urdu = sum(r["urdu_script"] for r in all_rows)
    print(f"Replies in Urdu script: {urdu}/{len(all_rows)}")
    print(f"Tool calls refused by the code (guardrails): {refusals}")
    print("Seconds per turn:      median   slowest")
    for stage in ("stt", "agent", "tts", "total"):
        values = [r[stage] for r in all_rows]
        print(f"  {stage:6}             {statistics.median(values):6.1f}   {max(values):7.1f}")


if __name__ == "__main__":
    main()