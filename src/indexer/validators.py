"""Check digits for the identifiers Italian business documents are keyed on.

A leaf module. Extraction -- by regex or by an LLM -- produces candidates; these
functions decide which candidates are real. That is the difference between an
index that links a document to customer C001 because its VAT number *is*
C001's, and one that links it because eleven digits appeared near the word
"IVA". Each identifier carries a checksum precisely so that a typo or an OCR
error is detectable, and ignoring it throws that away.

``partita_iva``     11 digits, Luhn-style check digit.
``codice_fiscale``  16 characters with the control letter (omocodia included),
                    or 11 digits for companies (same check as a VAT number).
``iban``            ISO 13616 mod-97; Italian IBANs are 27 characters.
"""

from __future__ import annotations

import re

__all__ = [
    "normalize_iban",
    "normalize_piva",
    "valid_codice_fiscale",
    "valid_iban",
    "valid_piva",
]

_ODD = {
    "0": 1, "1": 0, "2": 5, "3": 7, "4": 9, "5": 13, "6": 15, "7": 17, "8": 19, "9": 21,
    "A": 1, "B": 0, "C": 5, "D": 7, "E": 9, "F": 13, "G": 15, "H": 17, "I": 19, "J": 21,
    "K": 2, "L": 4, "M": 18, "N": 20, "O": 11, "P": 3, "Q": 6, "R": 8, "S": 12, "T": 14,
    "U": 16, "V": 10, "W": 22, "X": 25, "Y": 24, "Z": 23,
}  # fmt: skip
_CF_SHAPE = re.compile(
    r"^[A-Z]{6}[0-9LMNPQRSTUV]{2}[ABCDEHLMPRST][0-9LMNPQRSTUV]{2}[A-Z][0-9LMNPQRSTUV]{3}[A-Z]$"
)
_IBAN_LENGTHS = {"IT": 27, "SM": 27, "DE": 22, "FR": 27, "ES": 24, "GB": 22, "CH": 21, "AT": 20}


def valid_piva(value: str) -> bool:
    """An Italian VAT number (partita IVA): 11 digits and a valid check digit.

    Accepts an ``IT`` prefix and spaces, as the number is often written.
    """
    digits = re.sub(r"\s+", "", value.upper()).removeprefix("IT")
    if not re.fullmatch(r"\d{11}", digits) or digits == "0" * 11:
        return False
    d = [int(c) for c in digits]
    odd = sum(d[0:10:2])
    even = sum(2 * v if 2 * v <= 9 else 2 * v - 9 for v in d[1:10:2])
    return (10 - (odd + even) % 10) % 10 == d[10]


def normalize_piva(value: str) -> str:
    """``IT`` + 11 digits: the form FatturaPA writes, so both sources join."""
    return "IT" + re.sub(r"\s+", "", value.upper()).removeprefix("IT")


def valid_codice_fiscale(value: str) -> bool:
    """A codice fiscale: a person's 16-character code, or a company's 11 digits."""
    cf = re.sub(r"\s+", "", value.upper())
    if len(cf) == 11:
        return valid_piva(cf)
    if not _CF_SHAPE.match(cf):
        return False
    total = sum(_ODD[c] if i % 2 == 0 else _even(c) for i, c in enumerate(cf[:15]))
    return chr(ord("A") + total % 26) == cf[15]


def _even(c: str) -> int:
    return int(c) if c.isdigit() else ord(c) - ord("A")


def valid_iban(value: str) -> bool:
    """An IBAN whose mod-97 check holds (and, for known countries, whose length does)."""
    iban = normalize_iban(value)
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}", iban):
        return False
    expected = _IBAN_LENGTHS.get(iban[:2])
    if expected is not None and len(iban) != expected:
        return False
    rearranged = iban[4:] + iban[:4]
    return int("".join(str(int(ch, 36)) for ch in rearranged)) % 97 == 1


def normalize_iban(value: str) -> str:
    return re.sub(r"[\s-]+", "", value.upper())
