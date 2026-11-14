"""The brain: Gemini with bank tools. Takes the caller's Urdu text, replies in Urdu script the TTS can say."""
import functools
import os
import sys
import time
from datetime import date

from dotenv import load_dotenv
from google import genai
from google.genai import types

from urdu_voice.bank import Bank, BankError, Call
from urdu_voice.speech_text import speakable

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
RETRYABLE = ("429", "500", "503", "ReadError", "ConnectError", "RemoteProtocolError", "Timeout")
GREETING = "السلام علیکم، بینک ہیلپ لائن میں خوش آمدید۔ میں آپ کی کیا مدد کر سکتا ہوں؟"

# Digit words as Whisper writes them, including its common misspelling سفر for صفر (zero).
SPOKEN_DIGITS = {"صفر": "0", "سفر": "0", "زیرو": "0", "ایک": "1", "دو": "2", "تین": "3", "چار": "4",
                 "پانچ": "5", "چھ": "6", "چھے": "6", "سات": "7", "آٹھ": "8", "نو": "9"}

SYSTEM_PROMPT = f"""You are the phone assistant of a Pakistani bank's helpline.
Callers speak Urdu, often mixed with English words. Their words come from speech recognition, so expect small spelling mistakes.

How you speak (your reply is turned into speech):
- Reply only in Urdu script. Never Roman Urdu, never Hindi or Devanagari, never English sentences.
- Write English banking words in Urdu script: اکاؤنٹ، بیلنس، کارڈ، اے ٹی ایم، ٹرانزیکشن، کمپلینٹ، بلاک، برانچ، آفیسر.
- Keep it short: one or two sentences. No lists, no markdown, no emojis.
- Write numbers as digits, like 25000 روپے or کارڈ 4521. Put روپے after every amount.
- Be polite and use آپ. Speak about yourself in the masculine form (سکتا ہوں).

Rules:
- Before giving any account information or doing any account action, verify the caller with verify_caller.
  Ask for the account number and the last 4 digits of their CNIC (شناختی کارڈ).
- Never guess or make up balances, numbers or dates. Only say what a tool returned.
- Blocking a card has two steps: block_card, then read the last 4 digits back and ask the caller to confirm,
  then confirm_block_card only after they say yes.
- If the caller didn't make a transaction or money was wrongly deducted, file a DisputedTransaction complaint and tell them the complaint number.
- If verification fails, read back the account number you used so the caller can correct it.
- If a tool returns an error, follow what the error says.
- Transfer to an officer only when the caller asks for a person, the call is locked after failed verification,
  or they want something your tools can't do. Don't transfer just because the caller isn't verified yet: ask again.
- Today is {date.today():%A %d %B %Y}."""

def numbers_from_speech(text: str) -> str:
    """'ایک صفر چار سات تین تین آٹھ دو' -> '10473382'. Spoken account, card and CNIC numbers become digits
    in code, so the model doesn't have to assemble them. Only runs of 3+ digit words: 'دو دن' stays as it is."""
    words, out, run = text.split(), [], []
    for word in words + [""]:
        if word in SPOKEN_DIGITS:
            run.append(word)
            continue
        out.append("".join(SPOKEN_DIGITS[w] for w in run) if len(run) >= 3 else " ".join(run))
        out.append(word)
        run = []
    return " ".join(w for w in out if w)

def make_tools(call: Call, trace: list[dict], state: dict) -> list:
    """Gemini's tools, all bound to this one call. The model never sees or chooses a session.
    state["turn"] counts caller messages; state["pending_block"] remembers a block waiting for a yes."""

    def tool(fn):
        @functools.wraps(fn)  # keeps the name, docstring and type hints Gemini reads
        def wrapper(*args, **kwargs):
            try:
                result = fn(*args, **kwargs)
            except BankError as e:  # a refusal goes back to the model as text, not a crash
                result = {"error": str(e)}
            trace.append({"tool": fn.__name__, "args": kwargs, "result": result})
            return result
        return wrapper

    @tool
    def verify_caller(account_number: str, cnic_last4: str) -> dict:
        """Check the caller's identity. Needed before any account information or action.

        Args:
            account_number: The caller's 8-digit account number, digits only.
            cnic_last4: The last 4 digits of the caller's CNIC (national identity card).
        """
        return call.verify(account_number, cnic_last4)

    @tool
    def get_balance() -> dict:
        """Get the verified caller's account balance in PKR."""
        return call.balance()

    @tool
    def get_recent_transactions(count: int) -> dict:
        """Get the verified caller's most recent transactions, newest first. Negative amount = money out.

        Args:
            count: How many transactions to return, 1 to 10.
        """
        return {"transactions": call.transactions(count)}

    @tool
    def list_cards() -> dict:
        """List the verified caller's cards with their last 4 digits, type and status."""
        return {"cards": call.cards()}

    @tool
    def block_card(card_last4: str, reason: str) -> dict:
        """Step 1 of blocking a card: prepares the block. It does NOT block yet.

        Args:
            card_last4: The last 4 digits of the card to block.
            reason: Why, in English, e.g. "lost", "stolen", "suspicious transaction".
        """
        own_cards = [card["last4"] for card in call.cards()]  # also refuses if not verified
        if card_last4 not in own_cards:
            return {"error": f"No card ending in {card_last4} on this account. Cards: {', '.join(own_cards)}."}
        state["pending_block"] = {"last4": card_last4, "reason": reason, "turn": state["turn"]}
        return {"status": "not blocked yet, waiting for the caller",
                "next": f"Do not call any more tools now. Reply to the caller: read {card_last4} back and ask "
                        "them to confirm. After they say yes, call confirm_block_card."}

    @tool
    def confirm_block_card(card_last4: str) -> dict:
        """Step 2 of blocking a card: blocks it, after the caller said yes to the read-back.

        Args:
            card_last4: The last 4 digits the caller confirmed.
        """
        pending = state["pending_block"]
        if pending is None or pending["last4"] != card_last4:
            return {"error": f"No block prepared for card {card_last4}. Call block_card first."}
        if pending["turn"] == state["turn"]:  # the caller hasn't spoken since the read-back
            return {"error": "The caller has not confirmed yet. Read the 4 digits back and wait for their answer."}
        state["pending_block"] = None
        return call.block_card(card_last4, pending["reason"])

    @tool
    def file_complaint(category: str, description: str) -> dict:
        """File a complaint for the verified caller and get a complaint number.

        Args:
            category: One of DisputedTransaction, CardIssue, SalaryNotReceived, Other.
            description: A short summary of the problem in English, with amounts and dates.
        """
        return call.file_complaint(category, description)

    @tool
    def transfer_to_officer(reason: str) -> dict:
        """Transfer the call to a human customer service officer. Works without verification.

        Args:
            reason: Why, in English.
        """
        return call.handoff(reason)

    @tool
    def get_branch_hours() -> dict:
        """Get the bank branches' opening hours."""
        return call.branch_hours()

    return [verify_caller, get_balance, get_recent_transactions, list_cards, block_card,
            confirm_block_card, file_complaint, transfer_to_officer, get_branch_hours]


class Agent:
    """One conversation. Gemini decides which tools to call; the SDK runs them and sends the results back."""

    def __init__(self, call: Call):
        self.trace: list[dict] = []
        self.state = {"turn": 0, "pending_block": None}
        self.client = genai.Client()  # keep a reference: a client that gets garbage-collected closes its connection
        self.chat = self.client.chats.create(
            model=MODEL,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=make_tools(call, self.trace, self.state),
                temperature=0,
            ),
        )

    def reply(self, caller_text: str) -> str:
        self.state["turn"] += 1
        caller_text = numbers_from_speech(caller_text)
        for attempt in range(5):
            try:
                return (self.chat.send_message(caller_text).text or "").strip()
            except Exception as e:  # rate limit, overloaded or network: wait and retry
                if attempt == 4 or not any(code in f"{type(e).__name__} {e}" for code in RETRYABLE):
                    raise
                print(f"   (retrying in {10 * 2 ** attempt}s: {type(e).__name__} {str(e)[:80]})")
                time.sleep(10 * 2 ** attempt)


def talk(agent: Agent, caller_text: str) -> None:
    tools_before = len(agent.trace)
    start = time.perf_counter()
    text = agent.reply(caller_text)
    seconds = time.perf_counter() - start
    print(f"\nCaller: {caller_text}")
    for step in agent.trace[tools_before:]:
        print(f"   [tool] {step['tool']}({step['args']}) -> {step['result']}")
    print(f"Agent ({seconds:.1f}s): {text}")
    print(f"   spoken as: {speakable(text)}")


DEMO = [
    "السلام علیکم، میرا اے ٹی ایم کارڈ گم ہو گیا ہے، اسے بلاک کر دیں",
    "میرا اکاؤنٹ نمبر 10473382 ہے اور شناختی کارڈ کے آخری چار ہندسے 4321 ہیں",
    "جی ہاں، 4521 والا ڈیبٹ کارڈ، بلاک کر دیں",
    "اور میرا بیلنس کتنا ہے؟",
    "کل میں نے اے ٹی ایم سے پندرہ ہزار نکالے تھے لیکن اکاؤنٹ سے پچیس ہزار کٹ گئے",
]

if __name__ == "__main__":
    # python -m urdu_voice.agent         -> a scripted demo call
    # python -m urdu_voice.agent --chat  -> type your own messages (empty line to stop)
    sys.stdout.reconfigure(encoding="utf-8")
    bank = Bank()
    agent = Agent(bank.new_call())
    print(f"Agent: {GREETING}")
    if "--chat" in sys.argv:
        sys.stdin.reconfigure(encoding="utf-8")
        while caller_text := input("\nYou: ").strip():
            talk(agent, caller_text)
    else:
        for caller_text in DEMO:
            talk(agent, caller_text)
    print("\nBank action log:")
    for row in bank.actions:
        print(" ", row)