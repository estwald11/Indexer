"""Numbers and dates as people write them, in Italian and English.

The regression that motivated this: the field extractor turned "1.250,00" into
1.25 and "1,5" into 15.0, and dropped "30/06/2025". Every case below is one a
real Italian invoice, contract or email contains.
"""

from __future__ import annotations

from datetime import date

import pytest

from indexer.impls.enrich import _coerce
from indexer.normalize import parse_bool, parse_date, parse_int, parse_number

NBSP = chr(0xA0)  # typeset amounts group thousands with a no-break space


class TestNumbers:
    @pytest.mark.parametrize(
        ("raw", "locale", "want"),
        [
            ("1.250,00", "it", 1250.0),
            ("1.250,00", "auto", 1250.0),
            ("1,250.00", "en", 1250.0),
            ("1,250.00", "auto", 1250.0),
            ("12,5", "it", 12.5),
            ("12,5", "auto", 12.5),
            ("1.250", "it", 1250.0),
            ("1.250", "auto", 1250.0),
            ("€ 1.234.567,89", "it", 1234567.89),
            ("EUR 99,90", "it", 99.9),
            ("1.250,00 euro", "it", 1250.0),
            ("(1.250,00)", "it", -1250.0),
            ("-3,5", "it", -3.5),
            (f"1{NBSP}250,00", "it", 1250.0),
            ("1'250.00", "en", 1250.0),
            ("0,500", "it", 0.5),
            ("100", "auto", 100.0),
        ],
    )
    def test_parses(self, raw: str, locale: str, want: float) -> None:
        assert parse_number(raw, locale) == pytest.approx(want)

    @pytest.mark.parametrize(
        ("raw", "locale"),
        [
            ("12,5", "en"),  # not 125: a comma group of one digit is not grouping
            ("12.5", "it"),
            ("1.2.3", "auto"),
            ("1.250,00", "en"),
            ("abc", "auto"),
            ("", "auto"),
            ("12-5", "auto"),
        ],
    )
    def test_refuses_rather_than_guesses(self, raw: str, locale: str) -> None:
        assert parse_number(raw, locale) is None

    def test_ints_must_be_integral(self) -> None:
        assert parse_int("1.000", "it") == 1000
        assert parse_int("1.000,50", "it") is None

    def test_the_documented_ambiguity_of_auto(self) -> None:
        # "0,500" is a half in Italian and five hundred in English. Auto reads
        # a three-digit group as thousands; a corpus with a known locale says so.
        assert parse_number("0,500", "auto") == 500.0
        assert parse_number("0,500", "it") == 0.5


class TestDates:
    @pytest.mark.parametrize(
        ("raw", "want"),
        [
            ("2025-06-30", date(2025, 6, 30)),
            ("2025-06-30T14:05:00", date(2025, 6, 30)),
            ("30/06/2025", date(2025, 6, 30)),
            ("30.06.2025", date(2025, 6, 30)),
            ("30-06-2025", date(2025, 6, 30)),
            ("30/06/25", date(2025, 6, 30)),
            ("2025/06/30", date(2025, 6, 30)),
            ("30 giugno 2025", date(2025, 6, 30)),
            ("30 Giugno 2025", date(2025, 6, 30)),
            ("1° marzo 2024", date(2024, 3, 1)),
            ("lunedì 3 marzo 2025", date(2025, 3, 3)),
            ("3 mar 2025", date(2025, 3, 3)),
            ("June 30, 2025", date(2025, 6, 30)),
            ("30 June 2025", date(2025, 6, 30)),
        ],
    )
    def test_parses(self, raw: str, want: date) -> None:
        assert parse_date(raw) == want

    def test_order_decides_numeric_dates(self) -> None:
        assert parse_date("04/05/2025") == date(2025, 5, 4)
        assert parse_date("04/05/2025", order="mdy") == date(2025, 4, 5)

    @pytest.mark.parametrize("raw", ["31/02/2025", "13/13/2025", "tomorrow", "giugno", ""])
    def test_impossible_or_partial_dates_are_none(self, raw: str) -> None:
        assert parse_date(raw) is None


class TestBooleans:
    @pytest.mark.parametrize(
        ("raw", "want"),
        [
            ("sì", True),
            ("Si", True),
            ("vero", True),
            ("no", False),
            ("falso", False),
            ("forse", None),
        ],
    )
    def test_italian_and_english(self, raw: str, want: bool | None) -> None:
        assert parse_bool(raw) is want


class TestFieldExtractionUsesTheLocale:
    def test_regression_italian_amount_is_not_a_thousandth(self) -> None:
        assert _coerce("1.250,00", "float") == 1250.0
        assert _coerce("1.250,00", "float", locale="it") == 1250.0
        assert _coerce("30/06/2025", "date") == date(2025, 6, 30)
        assert _coerce("1.000", "int", locale="it") == 1000
        assert _coerce("sì", "bool") is True
