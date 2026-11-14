"""A fake bank for the voice agent: made-up customers, and one Call object per phone call.

The safety rules live here in code, not in the AI's prompt:
- no account data until the caller is verified,
- the call locks after 3 wrong verification attempts,
- a caller can only block their own cards.
"""
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta

MAX_VERIFY_ATTEMPTS = 3
COMPLAINT_CATEGORIES = ("DisputedTransaction", "CardIssue", "SalaryNotReceived", "Other")
BRANCH_HOURS = {"monday_to_friday": "09:00-17:00", "saturday": "09:00-13:00", "sunday": "closed"}


class BankError(Exception):
    """A refusal with a message the agent can read and act on (like an HTTP 4xx with a reason)."""


@dataclass
class Card:
    last4: str
    type: str  # Debit or Credit
    status: str = "Active"


@dataclass
class Customer:
    name: str
    account_number: str
    cnic_last4: str
    balance: int
    cards: list[Card]
    transactions: list[tuple[date, str, int]]  # (date, description, amount); negative = money out


def made_up_customers() -> dict[str, Customer]:
    today = date.today()
    ago = lambda days: today - timedelta(days=days)
    customers = [
        Customer("احمد علی", "10473382", "4321", 25_000,
                 [Card("4521", "Debit"), Card("7788", "Credit")],
                 [(ago(1), "ATM withdrawal", -25_000), (ago(3), "Electricity bill", -6_200),
                  (ago(6), "Online transfer", -10_000), (ago(10), "Salary", 85_000),
                  (ago(12), "Mobile top-up", -500)]),
        Customer("سارہ خان", "20581947", "8765", 125_000,
                 [Card("3390", "Debit", status="Blocked")],
                 [(ago(2), "Salary", 125_000)]),
        Customer("عمران شاہ", "30962215", "1199", 3_500,
                 [Card("6614", "Debit")],
                 [(ago(4), "Card payment", -3_500)]),
    ]
    return {c.account_number: c for c in customers}


class Bank:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Every test case starts from the same data."""
        self.customers = made_up_customers()
        self.complaints: list[dict] = []
        self.actions: list[dict] = []  # what the agent actually did, for evaluation
        self._next_complaint = 58213

    def new_call(self) -> "Call":
        return Call(self)


@dataclass
class Call:
    """One phone call. The agent's code holds this object; the AI never sees or picks an id."""
    bank: Bank
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    customer: Customer | None = None
    failed_attempts: int = 0

    # ---------- always allowed ----------

    def verify(self, account_number: str, cnic_last4: str) -> dict:
        try:
            if self.failed_attempts >= MAX_VERIFY_ATTEMPTS:
                raise BankError("Too many failed attempts. Transfer the caller to an officer.")
            customer = self.bank.customers.get(account_number)
            if customer is None or customer.cnic_last4 != cnic_last4:
                self.failed_attempts += 1
                left = MAX_VERIFY_ATTEMPTS - self.failed_attempts
                raise BankError(f"Account number and CNIC digits don't match. Attempts left: {left}.")
            self.customer = customer
            self._log("verify", account_number, "ok")
            return {"verified": True, "name": customer.name}
        except BankError as e:
            self._log("verify", account_number, str(e))
            raise

    def handoff(self, reason: str) -> dict:
        self._log("handoff", reason, "ok")
        return {"message": "Transferred to a customer service officer."}

    def branch_hours(self) -> dict:
        return BRANCH_HOURS

    # ---------- only after verification ----------

    def balance(self) -> dict:
        return {"balance": self._verified().balance, "currency": "PKR"}

    def transactions(self, count: int = 5) -> list[dict]:
        count = max(1, min(count, 10))
        return [{"date": d.isoformat(), "description": text, "amount": amount}
                for d, text, amount in self._verified().transactions[:count]]

    def cards(self) -> list[dict]:
        return [vars(card) for card in self._verified().cards]

    def block_card(self, last4: str, reason: str) -> dict:
        try:
            customer = self._verified()
            card = next((c for c in customer.cards if c.last4 == last4), None)
            if card is None:
                own = ", ".join(c.last4 for c in customer.cards)
                raise BankError(f"No card ending in {last4} on this account. Cards: {own}.")
            if card.status == "Blocked":
                raise BankError(f"The card ending in {last4} is already blocked.")
            card.status = "Blocked"
            self._log("block_card", last4, "ok")
            return vars(card)
        except BankError as e:
            self._log("block_card", last4, str(e))
            raise

    def file_complaint(self, category: str, description: str) -> dict:
        try:
            customer = self._verified()
            if category not in COMPLAINT_CATEGORIES:
                raise BankError(f"Unknown category. Use one of: {', '.join(COMPLAINT_CATEGORIES)}.")
            number = str(self.bank._next_complaint)
            self.bank._next_complaint += 1
            self.bank.complaints.append({"number": number, "account": customer.account_number,
                                         "category": category, "description": description})
            self._log("complaint", category, "ok")
            return {"complaint_number": number}
        except BankError as e:
            self._log("complaint", category, str(e))
            raise

    # ---------- the guard ----------

    def _verified(self) -> Customer:
        """Like [Authorize]: every account method goes through here first."""
        if self.customer is None:
            raise BankError("The caller is not verified yet. "
                            "Ask for the account number and the last 4 digits of their CNIC.")
        return self.customer

    def _log(self, action: str, detail: str, result: str) -> None:
        self.bank.actions.append({"call": self.id, "action": action, "detail": detail, "result": result})


if __name__ == "__main__":
    # Quick self-check: python -m urdu_voice.bank
    bank = Bank()
    call = bank.new_call()

    def attempt(label, fn):
        try:
            print(f"{label}: {fn()}")
        except BankError as e:
            print(f"{label}: refused -> {e}")

    attempt("balance before verify", call.balance)
    attempt("wrong CNIC", lambda: call.verify("10473382", "0000"))
    attempt("right details", lambda: call.verify("10473382", "4321"))
    attempt("balance", call.balance)
    attempt("last 2 transactions", lambda: call.transactions(2))
    attempt("block someone else's card", lambda: call.block_card("3390", "lost"))
    attempt("block own card", lambda: call.block_card("4521", "lost"))
    attempt("block it again", lambda: call.block_card("4521", "lost"))
    attempt("complaint", lambda: call.file_complaint("DisputedTransaction", "15000 withdrew, 25000 deducted"))

    locked = bank.new_call()
    for _ in range(3):
        attempt("guessing", lambda: locked.verify("10473382", "1111"))
    attempt("right details after lock", lambda: locked.verify("10473382", "4321"))
    attempt("handoff without verify", lambda: locked.handoff("caller locked out"))

    print("\nAction log:")
    for row in bank.actions:
        print(" ", row)