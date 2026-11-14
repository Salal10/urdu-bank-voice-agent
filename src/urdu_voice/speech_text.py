"""Turn the agent's reply into text a text-to-speech model can say: numbers become Urdu words."""
import re

DIGITS = ["صفر", "ایک", "دو", "تین", "چار", "پانچ", "چھ", "سات", "آٹھ", "نو"]
UNDER_100 = (
    "صفر ایک دو تین چار پانچ چھ سات آٹھ نو "
    "دس گیارہ بارہ تیرہ چودہ پندرہ سولہ سترہ اٹھارہ انیس "
    "بیس اکیس بائیس تیئس چوبیس پچیس چھبیس ستائیس اٹھائیس انتیس "
    "تیس اکتیس بتیس تینتیس چونتیس پینتیس چھتیس سینتیس اڑتیس انتالیس "
    "چالیس اکتالیس بیالیس تینتالیس چوالیس پینتالیس چھیالیس سینتالیس اڑتالیس انچاس "
    "پچاس اکاون باون ترپن چون پچپن چھپن ستاون اٹھاون انسٹھ "
    "ساٹھ اکسٹھ باسٹھ تریسٹھ چونسٹھ پینسٹھ چھیاسٹھ سڑسٹھ اڑسٹھ انہتر "
    "ستر اکہتر بہتر تہتر چوہتر پچہتر چھہتر ستتر اٹھہتر اناسی "
    "اسی اکاسی بیاسی تراسی چوراسی پچاسی چھیاسی ستاسی اٹھاسی نواسی "
    "نوے اکانوے بانوے ترانوے چورانوے پچانوے چھیانوے ستانوے اٹھانوے ننانوے"
).split()
SCALES = [(10_000_000, "کروڑ"), (100_000, "لاکھ"), (1000, "ہزار"), (100, "سو")]  # Pakistani number system
AMOUNT_WORDS = ("روپے", "روپیہ", "روپیے")


def amount_in_words(n: int) -> str:
    """125000 -> 'ایک لاکھ پچیس ہزار'."""
    if n < 100:
        return UNDER_100[n]
    for size, name in SCALES:
        if n >= size:
            head, rest = divmod(n, size)
            words = f"{amount_in_words(head)} {name}"
            return f"{words} {amount_in_words(rest)}" if rest else words
    raise ValueError(n)


def digit_by_digit(number: str) -> str:
    """'4521' -> 'چار پانچ دو ایک', the way card and account numbers are read on a call."""
    return " ".join(DIGITS[int(d)] for d in number)


NUMBER = re.compile(r"\d[\d,]*\d|\d")


def say_number(match: re.Match, text: str) -> str:
    """Amounts (a number before روپے) and short numbers are read as amounts; longer ones digit by digit."""
    number = match.group().replace(",", "")
    after = text[match.end():].lstrip()
    if after.startswith(AMOUNT_WORDS) or len(number) <= 2:
        return amount_in_words(int(number))
    return digit_by_digit(number)


def speakable(text: str) -> str:
    return NUMBER.sub(lambda m: say_number(m, text), text)


def spoken_numbers(text: str) -> list[str]:
    """How each number in the text should sound, e.g. ['پچیس ہزار']."""
    return [say_number(m, text) for m in NUMBER.finditer(text)]


def unspeakable_words(text: str) -> list[str]:
    """Latin-script words the Urdu voice can't say (it silently skips them)."""
    return re.findall(r"[A-Za-z]+", text)