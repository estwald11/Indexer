"""Locale-aware parsing of numbers, dates and booleans written by people.

A leaf module: extractors, the router and validators all turn text into typed
values, and they must agree on how.

The failure this replaces was silent and three orders of magnitude wide. The
field extractor stripped commas and called ``float``: "1.250,00" -- one
thousand two hundred and fifty euros, as every Italian invoice writes it --
became ``1.25``, and "1,5" became ``15.0``. Dates were ISO-only, so
"30/06/2025" was dropped. A structured question over those fields then returns
a confident wrong answer, which is worse than no answer.

Separators
----------
``it``    "." groups thousands, "," is the decimal mark: 1.250,00 -> 1250.0
``en``    "," groups thousands, "." is the decimal mark: 1,250.00 -> 1250.0
``auto``  When both marks appear, the last one is the decimal mark. When only
          one appears, a single mark followed by exactly three digits is read
          as a thousands separator (1.250 and 1,250 are both 1250) and anything
          else as a decimal mark (12,5 and 12.5 are both 12.5). That is right
          for nearly every amount and wrong for "0,500" meaning one half --
          set the locale explicitly when a corpus has one.
"""

from __future__ import annotations

import re
from datetime import date, datetime

__all__ = ["parse_bool", "parse_date", "parse_int", "parse_number"]

_CURRENCY = re.compile(r"(?i)(€|eur(?:o|os)?|usd|\$|£|gbp|chf|fr\.)")
# ``\s`` already matches the no-break and thin spaces that typeset amounts use.
_SPACES = re.compile(r"[\s']")
_NUMBER_BODY = re.compile(r"^[+-]?\d[\d.,]*$")


def parse_number(raw: str, locale: str = "auto") -> float | None:
    """A number as written in ``locale``, or None if it is not one.

    Currency marks, spaces (including the non-breaking and thin spaces that
    typeset amounts use) and apostrophe grouping ("1'250.00") are ignored.
    Accounting negatives in parentheses -- "(1.250,00)" -- are negative.
    """
    s = raw.strip()
    negative = s.startswith("(") and s.endswith(")")
    if negative:
        s = s[1:-1]
    s = _CURRENCY.sub("", s)
    s = _SPACES.sub("", s)
    if s.startswith("-"):
        negative = not negative
        s = s[1:]
    elif s.startswith("+"):
        s = s[1:]
    if not s or not _NUMBER_BODY.match(s):
        return None
    decimal = _decimal_mark(s, locale)
    if decimal is None:
        return None
    thousands = "," if decimal == "." else "."
    if decimal == "":
        # No decimal mark, so every mark present must be grouping -- one kind,
        # in groups of three. "12,5" is not 125 in any locale.
        marks = {c for c in s if c in ".,"}
        if len(marks) > 1 or (marks and not _valid_grouping(s, marks.pop())):
            return None
        body = s.replace(".", "").replace(",", "")
    else:
        whole, _, frac = s.rpartition(decimal)
        if thousands in frac or decimal in whole:
            return None
        if not _valid_grouping(whole, thousands):
            return None
        body = whole.replace(thousands, "") + "." + frac
    try:
        value = float(body)
    except ValueError:
        return None
    return -value if negative else value


def _decimal_mark(s: str, locale: str) -> str | None:
    """ "." or "," for the decimal mark, "" when the number has none."""
    has_dot, has_comma = "." in s, "," in s
    if locale == "it":
        return "," if has_comma else ""
    if locale == "en":
        return "." if has_dot else ""
    if locale != "auto":
        raise ValueError(f"locale must be it, en or auto, not {locale!r}")
    if has_dot and has_comma:
        return "." if s.rfind(".") > s.rfind(",") else ","
    mark = "." if has_dot else "," if has_comma else ""
    if not mark:
        return ""
    if s.count(mark) > 1:
        return ""  # 1.250.000 / 1,250,000: grouping only
    tail = s.rsplit(mark, 1)[1]
    return "" if len(tail) == 3 else mark


def _valid_grouping(whole: str, sep: str) -> bool:
    """Thousands groups, when present, must be groups of three."""
    if sep not in whole:
        return True
    head, *groups = whole.split(sep)
    return 1 <= len(head) <= 3 and all(len(g) == 3 for g in groups)


def parse_int(raw: str, locale: str = "auto") -> int | None:
    """An integer as written in ``locale``; None for non-integral values."""
    value = parse_number(raw, locale)
    if value is None or value != int(value):
        return None
    return int(value)


_MONTHS = {
    # Italian
    "gennaio": 1, "gen": 1, "febbraio": 2, "feb": 2, "marzo": 3, "mar": 3,
    "aprile": 4, "apr": 4, "maggio": 5, "mag": 5, "giugno": 6, "giu": 6,
    "luglio": 7, "lug": 7, "agosto": 8, "ago": 8, "settembre": 9, "set": 9,
    "sett": 9, "ottobre": 10, "ott": 10, "novembre": 11, "nov": 11,
    "dicembre": 12, "dic": 12,
    # English
    "january": 1, "jan": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9, "october": 10, "oct": 10,
    "november": 11, "december": 12, "dec": 12,
}  # fmt: skip

_ISO = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})(?:[t ].*)?$")
_NUMERIC = re.compile(r"^(\d{1,4})[/.\-](\d{1,2})[/.\-](\d{1,4})$")
_DAY_MONTH_YEAR = re.compile(
    r"^(?:(?:lun|mar|mer|gio|ven|sab|dom|mon|tue|wed|thu|fri|sat|sun)\w*,?\s+)?"
    r"(\d{1,2})(?:°|º|st|nd|rd|th)?\s+([a-zà-ù]+)\.?,?\s+(\d{2,4})$"
)
_MONTH_DAY_YEAR = re.compile(r"^([a-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})$")


def parse_date(raw: str, *, order: str = "dmy") -> date | None:
    """A calendar date as people write it, or None.

    ISO first (``2025-06-30``, with or without a time), then numeric dates in
    ``order`` -- ``dmy`` for Italy and most of Europe, ``mdy`` for the United
    States -- with "/", "-" or "." between the parts, then written months in
    Italian or English ("30 giugno 2025", "1° marzo 2024", "June 30, 2025").
    Two-digit years pivot at 70. An impossible date (31/02) is None, never a
    guess.
    """
    s = " ".join(raw.strip().lower().split())
    if not s:
        return None
    if order not in ("dmy", "mdy"):
        # Outside the try: a misconfigured order is a config error, not a date
        # that failed to parse.
        raise ValueError(f"order must be dmy or mdy, not {order!r}")
    try:
        if m := _ISO.match(s):
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if m := _NUMERIC.match(s):
            a, b, c = (int(g) for g in m.groups())
            if len(m.group(1)) == 4:  # 2025/06/30
                return date(a, b, c)
            day, month = (a, b) if order == "dmy" else (b, a)
            return date(_year(c, m.group(3)), month, day)
        if m := _DAY_MONTH_YEAR.match(s):
            named = _MONTHS.get(m.group(2))
            if named is None:
                return None
            return date(_year(int(m.group(3)), m.group(3)), named, int(m.group(1)))
        if m := _MONTH_DAY_YEAR.match(s):
            named = _MONTHS.get(m.group(1))
            if named is None:
                return None
            return date(int(m.group(3)), named, int(m.group(2)))
    except ValueError:
        return None
    return None


def _year(value: int, text: str) -> int:
    if len(text) == 2:
        return 2000 + value if value < 70 else 1900 + value
    return value


_TRUE = frozenset({"true", "yes", "y", "1", "sì", "si", "vero", "x"})
_FALSE = frozenset({"false", "no", "n", "0", "falso"})


def parse_bool(raw: str) -> bool | None:
    """Yes/no in English or Italian; None when it is neither."""
    s = raw.strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    return None


def parse_datetime(raw: str) -> datetime | None:
    """An ISO datetime, or None. Only ISO: a time of day has no safe locale guess."""
    try:
        return datetime.fromisoformat(raw.strip())
    except ValueError:
        return None
