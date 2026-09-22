"""Email parsers: RFC 822 messages (and PEC envelopes), and Outlook ``.msg``.

A message is indexed as what a reader sees: the subject as its heading, who
wrote to whom and when, the body -- without the quoted history of every reply
before it, which otherwise makes each message of a thread match every query
the thread matches -- and the names of its attachments. The attachments
themselves are documents of their own when the scanner expands emails.

Scanner and parser agree on the facts (``email_subject``, ``email_from`` ...)
through ``indexer.impls.containers.parse_email``; the parser adds a typed
``sent_date`` so "emails since March" is a date comparison, not a string one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any

from indexer.core.document import BlockKind, ParsedDocument, SourceDocument
from indexer.core.registry import register
from indexer.core.stages import StageContext
from indexer.impls.containers import EmailInfo, parse_email
from indexer.impls.parse import _Builder
from indexer.plugin import StageImpl, dataclass_params

__all__ = ["EmailParser", "MsgParser", "strip_quoted"]

_LABELS = {
    "it": {
        "from": "Da",
        "to": "A",
        "cc": "Cc",
        "date": "Data",
        "attachments": "Allegati",
        "no_subject": "(senza oggetto)",
        "pec": "Posta certificata",
    },
    "en": {
        "from": "From",
        "to": "To",
        "cc": "Cc",
        "date": "Date",
        "attachments": "Attachments",
        "no_subject": "(no subject)",
        "pec": "Certified email",
    },
}

#: Where a reply's quoted history begins, in the forms Italian and English
#: mail clients write it.
_QUOTE_MARKERS = re.compile(
    r"^(?:-{2,}\s*(?:original message|messaggio originale|forwarded message|"
    r"messaggio inoltrato)\s*-{2,}"
    r"|il giorno .{3,80} ha scritto:?"
    r"|on .{3,80} wrote:?"
    r"|(?:da|from):\s.+\n(?:inviato|sent|data|date):\s.+)",
    re.I | re.M,
)


def strip_quoted(body: str) -> tuple[str, bool]:
    """The message's own text, and whether quoted history was removed."""
    m = _QUOTE_MARKERS.search(body)
    own = body[: m.start()] if m else body
    lines = [ln for ln in own.splitlines() if not ln.lstrip().startswith(">")]
    stripped = "\n".join(lines).rstrip()
    if not stripped.strip():
        # A bare forward: the "quoted" part is the whole message, and it is
        # what the sender meant to send.
        return body, False
    return stripped, (m is not None or len(lines) != len(own.splitlines()))


@dataclass(frozen=True, slots=True)
class EmailParams:
    #: Remove the quoted history of replies and forwards.
    strip_quoted: bool = True
    #: Language of the header labels written into the text: it or en.
    labels: str = "en"


@register(
    "parse",
    "email",
    version="1",
    params_model=dataclass_params(EmailParams),
    summary="RFC 822 email and PEC: subject, participants, date, body without quoted "
    "history, attachment names. Standard library.",
)
def _make_email(params: dict[str, Any], **_: Any) -> EmailParser:
    return EmailParser(params)


class EmailParser(StageImpl):
    STAGE, IMPL, VERSION = "parse", "email", "1"

    def can_parse(self, doc: SourceDocument) -> float:
        return 0.95 if doc.media_type == "message/rfc822" else 0.0

    def parse(self, doc: SourceDocument, ctx: StageContext) -> ParsedDocument:
        info = parse_email(doc.load())
        return _render(doc, info, self._params)


def _render(doc: SourceDocument, info: EmailInfo, params: dict[str, Any]) -> ParsedDocument:
    labels = _LABELS.get(str(params.get("labels", "en")), _LABELS["en"])
    b = _Builder(doc.document_id, doc.source_uri)
    b.add(info.subject.strip() or labels["no_subject"], BlockKind.HEADING, level=1)
    header = [f"{labels['from']}: {info.sender}"] if info.sender else []
    if info.to:
        header.append(f"{labels['to']}: {', '.join(info.to)}")
    if info.cc:
        header.append(f"{labels['cc']}: {', '.join(info.cc)}")
    if info.date:
        header.append(f"{labels['date']}: {info.date}")
    if header:
        b.add("\n".join(header), BlockKind.PARAGRAPH, attrs={"role": "email_header"})
    if info.pec:
        facts = ", ".join(
            f"{k}: {v if not isinstance(v, list) else ', '.join(v)}" for k, v in info.pec.items()
        )
        b.add(f"{labels['pec']} -- {facts}", BlockKind.PARAGRAPH, attrs={"role": "pec"})
    body, quoted = (
        strip_quoted(info.body) if params.get("strip_quoted", True) else (info.body, False)
    )
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if para:
            b.add(para, BlockKind.PARAGRAPH)
    names = [
        a.name for a in info.attachments if a.name.lower() not in ("daticert.xml", "smime.p7s")
    ]
    if names:
        b.add(f"{labels['attachments']}: {', '.join(names)}", BlockKind.PARAGRAPH)

    meta: dict[str, Any] = dict(doc.metadata)
    added = {
        "email_subject": info.subject,
        "email_from": info.sender,
        "email_to": list(info.to),
        "email_cc": list(info.cc),
        "email_date": info.date,
        "email_message_id": info.message_id,
        "attachment_names": names,
        "quoted_removed": quoted,
    }
    sent = _sent_at(info.date)
    if sent is not None:
        added["sent_date"] = sent.date()
    if info.pec:
        added["pec"] = True
        added.update({f"pec_{k}": v for k, v in info.pec.items()})
    for k, v in added.items():
        if v not in ("", [], None):
            meta.setdefault(k, v)
    return b.finish(doc.content_hash, metadata=meta)


def _sent_at(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None


# --------------------------------------------------------------------------- #
# Outlook .msg                                                                 #
# --------------------------------------------------------------------------- #


@register(
    "parse",
    "msg",
    version="1",
    params_model=dataclass_params(EmailParams),
    summary="Outlook .msg via extract-msg (GPL-3.0: check it fits your distribution).",
    requires=("extract-msg",),
)
def _make_msg(params: dict[str, Any], **_: Any) -> MsgParser:
    return MsgParser(params)


class MsgParser(StageImpl):
    """Outlook messages, rendered exactly like RFC 822 ones.

    ``extract-msg`` is GPL-3.0. Using it inside a company is usually fine;
    shipping it inside a product may not be, which is why it is an explicit
    choice in config rather than a default.
    """

    STAGE, IMPL, VERSION = "parse", "msg", "1"

    def can_parse(self, doc: SourceDocument) -> float:
        return 0.95 if doc.media_type == "application/vnd.ms-outlook" else 0.0

    def parse(self, doc: SourceDocument, ctx: StageContext) -> ParsedDocument:
        try:
            import extract_msg
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("the msg parser needs extract-msg: pip install extract-msg") from exc
        from indexer.impls.containers import Attachment

        msg = extract_msg.openMsg(doc.load())
        try:
            attachments = tuple(
                Attachment(
                    name=str(getattr(a, "longFilename", None) or getattr(a, "shortFilename", "")),
                    media_type="application/octet-stream",
                    payload=b"",
                    index=i,
                )
                for i, a in enumerate(getattr(msg, "attachments", []) or [])
            )
            info = EmailInfo(
                subject=str(msg.subject or ""),
                sender=str(msg.sender or ""),
                to=tuple(a.strip() for a in str(msg.to or "").split(";") if a.strip()),
                cc=tuple(a.strip() for a in str(msg.cc or "").split(";") if a.strip()),
                date=str(msg.date or ""),
                message_id=str(getattr(msg, "messageId", "") or ""),
                body=str(msg.body or ""),
                attachments=attachments,
            )
        finally:
            close = getattr(msg, "close", None)
            if callable(close):
                close()
        return _render(doc, info, self._params)
