"""Containers an Italian company archive is full of: signed envelopes, email, zip.

Three formats hold most of the documents that matter in such an archive, and
each hides its content from a parser that only reads files:

``.p7m``   A CAdES envelope: the signed document (an invoice XML, a contract
           PDF) wrapped in CMS SignedData. The document is inside, as bytes,
           and the signers' certificates beside it. FatturaPA invoices travel
           this way, and so does every digitally signed contract.
``.eml``   A message, and its attachments -- which are usually the documents:
           the order, the quote, the signed contract. A PEC (posta elettronica
           certificata) is an ``.eml`` whose attachments are the original
           message (``postacert.eml``), the provider's certification data
           (``daticert.xml``) and a signature.
``.zip``   Folders of the above.

Everything here is standard library, including the CMS reader: a minimal BER
walker is enough to reach the encapsulated content and the signers' names, and
it keeps a dependency out of the path every signed document takes. The
signature is *not* verified -- that is a trust decision, not an indexing one,
and it is recorded as ``signature_verified: false`` rather than implied.
"""

from __future__ import annotations

import base64
import binascii
import email
import email.policy
import io
import mimetypes
import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Any, ClassVar
from xml.etree import ElementTree

__all__ = [
    "Attachment",
    "EmailInfo",
    "SignedContent",
    "html_to_text",
    "is_p7m",
    "media_type_for",
    "parse_daticert",
    "parse_email",
    "unwrap_p7m",
    "zip_members",
]

# --------------------------------------------------------------------------- #
# media types                                                                  #
# --------------------------------------------------------------------------- #

_EXTRA_TYPES = {
    ".md": "text/markdown",
    ".rst": "text/x-rst",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".yaml": "text/yaml",
    ".yml": "text/yaml",
    ".toml": "text/toml",
    ".cfg": "text/plain",
    ".in": "text/plain",
    ".eml": "message/rfc822",
    ".msg": "application/vnd.ms-outlook",
    ".p7m": "application/pkcs7-mime",
    ".xml": "application/xml",
    ".html": "text/html",
    ".htm": "text/html",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".zip": "application/zip",
}


def media_type_for(name: str) -> str:
    """The media type a file name implies. Deterministic across platforms --
    ``mimetypes`` alone reads the Windows registry, and a scan on another
    machine must not decide a different parser."""
    lower = name.lower()
    for ext, mt in _EXTRA_TYPES.items():
        if lower.endswith(ext):
            return mt
    guessed, _ = mimetypes.guess_type(name, strict=True)
    return guessed or "application/octet-stream"


# --------------------------------------------------------------------------- #
# CMS / CAdES (.p7m)                                                           #
# --------------------------------------------------------------------------- #

_OID_SIGNED_DATA = bytes.fromhex("2a864886f70d010702")  # 1.2.840.113549.1.7.2
_OID_COMMON_NAME = bytes.fromhex("550403")  # 2.5.4.3
_OID_SERIAL_NUMBER = bytes.fromhex("550405")  # 2.5.4.5 -- the fiscal code, in Italy


@dataclass(frozen=True, slots=True)
class SignedContent:
    content: bytes
    #: Common names from the certificates in the envelope, in order.
    signers: tuple[str, ...] = ()
    #: The subject serial number of those certificates: an Italian qualified
    #: certificate carries the holder's fiscal code there ("TINIT-...").
    signer_ids: tuple[str, ...] = ()
    #: How many envelopes were removed ("x.pdf.p7m.p7m" is two).
    layers: int = 1


@dataclass(slots=True)
class _Node:
    tag: int
    constructed: bool
    start: int  # content start
    end: int  # content end (exclusive)
    children: list[_Node] = field(default_factory=list)


class BERError(ValueError):
    pass


def _read_node(data: bytes, pos: int, depth: int = 0) -> tuple[_Node, int]:
    """One BER TLV at ``pos``, children parsed for constructed types.

    BER rather than DER because real envelopes use it: indefinite lengths and
    octet strings split into chunks are both common in .p7m files written by
    Italian signing software, and a DER-only reader rejects them.
    """
    if depth > 64:
        raise BERError("nesting too deep")
    if pos + 2 > len(data):
        raise BERError("truncated tag")
    first = data[pos]
    tag = first
    pos += 1
    if first & 0x1F == 0x1F:  # high tag number form
        while True:
            if pos >= len(data):
                raise BERError("truncated tag")
            b = data[pos]
            pos += 1
            tag = (tag << 8) | b
            if not b & 0x80:
                break
    constructed = bool(first & 0x20)
    if pos >= len(data):
        raise BERError("truncated length")
    length_byte = data[pos]
    pos += 1
    if length_byte == 0x80:  # indefinite
        if not constructed:
            raise BERError("indefinite length on a primitive")
        node = _Node(tag, True, pos, pos)
        while True:
            if data[pos : pos + 2] == b"\x00\x00":
                node.end = pos
                return node, pos + 2
            child, pos = _read_node(data, pos, depth + 1)
            node.children.append(child)
    if length_byte & 0x80:
        n = length_byte & 0x7F
        if n > 8 or pos + n > len(data):
            raise BERError("bad length")
        length = int.from_bytes(data[pos : pos + n], "big")
        pos += n
    else:
        length = length_byte
    end = pos + length
    if end > len(data):
        raise BERError("content runs past the end")
    node = _Node(tag, constructed, pos, end)
    if constructed:
        p = pos
        while p < end:
            child, p = _read_node(data, p, depth + 1)
            node.children.append(child)
    return node, end


def _octets(data: bytes, node: _Node) -> bytes:
    """An OCTET STRING's bytes, joining the chunks of a constructed one."""
    if not node.constructed:
        return data[node.start : node.end]
    return b"".join(_octets(data, c) for c in node.children)


def _der_payload(raw: bytes) -> bytes:
    """The DER bytes of an envelope, which may be stored base64-encoded."""
    stripped = raw.strip()
    if stripped[:1] == b"\x30":
        return raw
    text = re.sub(rb"-----[^-]+-----", b"", stripped)
    try:
        decoded = base64.b64decode(re.sub(rb"\s+", b"", text), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise BERError("neither DER nor base64") from exc
    if decoded[:1] != b"\x30":
        raise BERError("not a CMS structure")
    return decoded


def is_p7m(raw: bytes) -> bool:
    try:
        root, _ = _read_node(_der_payload(raw), 0)
    except (BERError, IndexError):
        return False
    return bool(root.children) and _is_signed_data_oid(_der_payload(raw), root.children[0])


def _is_signed_data_oid(data: bytes, node: _Node) -> bool:
    return node.tag == 0x06 and data[node.start : node.end] == _OID_SIGNED_DATA


def unwrap_p7m(raw: bytes, *, max_layers: int = 4) -> SignedContent:
    """The signed document inside a CAdES envelope, and who signed it.

    Unwraps nested envelopes (a document countersigned is enveloped twice).
    Raises ``ValueError`` for anything that is not an attached-content
    SignedData -- a detached signature (a ``.p7s`` next to its document) has
    nothing to unwrap.
    """
    signers: list[str] = []
    ids: list[str] = []
    content = raw
    layers = 0
    while layers < max_layers:
        try:
            data = _der_payload(content)
            root, _ = _read_node(data, 0)
        except (BERError, IndexError) as exc:
            if layers:
                break
            raise ValueError(f"not a CMS envelope: {exc}") from exc
        if len(root.children) < 2 or not _is_signed_data_oid(data, root.children[0]):
            if layers:
                break
            raise ValueError("not a CMS SignedData envelope")
        explicit = root.children[1]
        signed_data = explicit.children[0] if explicit.children else None
        if signed_data is None or len(signed_data.children) < 3:
            raise ValueError("malformed SignedData")
        encap = signed_data.children[2]
        if len(encap.children) < 2 or not encap.children[1].children:
            raise ValueError("detached signature: the envelope carries no content")
        content = _octets(data, encap.children[1].children[0])
        for child in signed_data.children[3:]:
            if child.tag == 0xA0:  # certificates [0] IMPLICIT
                for cert in child.children:
                    cn, serial = _subject_names(data, cert)
                    if cn and cn not in signers:
                        signers.append(cn)
                    if serial and serial not in ids:
                        ids.append(serial)
        layers += 1
        if not is_p7m(content):
            break
    return SignedContent(content, tuple(signers), tuple(ids), layers)


def _subject_names(data: bytes, cert: _Node) -> tuple[str, str]:
    """(commonName, serialNumber) from a certificate's subject, best effort."""
    try:
        tbs = cert.children[0]
        fields = tbs.children
        i = 1 if fields and fields[0].tag == 0xA0 else 0  # optional [0] version
        subject = fields[i + 4]
    except IndexError:
        return "", ""
    cn = serial = ""
    for rdn in subject.children:
        for atv in rdn.children:
            if len(atv.children) < 2:
                continue
            oid = data[atv.children[0].start : atv.children[0].end]
            value = data[atv.children[1].start : atv.children[1].end]
            text = _decode_string(value, atv.children[1].tag)
            if oid == _OID_COMMON_NAME and not cn:
                cn = text
            elif oid == _OID_SERIAL_NUMBER and not serial:
                serial = text
    return cn, serial


def _decode_string(value: bytes, tag: int) -> str:
    if tag == 0x1E:  # BMPString
        return value.decode("utf-16-be", errors="replace")
    return value.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# email and PEC                                                                #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Attachment:
    name: str
    media_type: str
    payload: bytes
    #: The attachment's position among the message's attachments -- its
    #: address, stable across re-reads of the same bytes.
    index: int


@dataclass(frozen=True, slots=True)
class EmailInfo:
    subject: str
    sender: str
    to: tuple[str, ...]
    cc: tuple[str, ...]
    date: str
    message_id: str
    body: str
    attachments: tuple[Attachment, ...]
    #: Present when the message is a PEC envelope: the provider's
    #: certification data (``daticert.xml``), parsed.
    pec: dict[str, Any] | None = None

    @property
    def is_pec(self) -> bool:
        return self.pec is not None


def parse_email(raw: bytes) -> EmailInfo:
    """Headers, a plain-text body and the attachments of an RFC 822 message.

    The body prefers ``text/plain`` and falls back to converting HTML. A PEC
    envelope is recognised by its ``daticert.xml``; the certification data is
    parsed, and the envelope's own body -- the provider's boilerplate -- is
    kept, since it states who sent what to whom and when.
    """
    msg = email.message_from_bytes(raw, policy=email.policy.default)
    assert isinstance(msg, EmailMessage)
    attachments: list[Attachment] = []
    plain: list[str] = []
    html: list[str] = []
    for part in _parts(msg):
        if part.is_multipart() and part.get_content_type() != "message/rfc822":
            continue
        disposition = part.get_content_disposition()
        filename = part.get_filename()
        ctype = part.get_content_type()
        if filename or disposition == "attachment" or ctype == "message/rfc822":
            payload = _payload_bytes(part)
            if payload is None:
                continue
            name = filename or (
                "message.eml" if ctype == "message/rfc822" else f"attachment-{len(attachments)}"
            )
            attachments.append(
                Attachment(
                    name=name,
                    media_type=media_type_for(name) if "." in name else ctype,
                    payload=payload,
                    index=len(attachments),
                )
            )
            continue
        if ctype == "text/plain":
            plain.append(_text_of(part))
        elif ctype == "text/html":
            html.append(_text_of(part))
    body = "\n\n".join(t.strip() for t in plain if t.strip())
    if not body and html:
        body = "\n\n".join(html_to_text(h) for h in html if h.strip())
    pec = None
    for a in attachments:
        if a.name.lower() == "daticert.xml":
            pec = parse_daticert(a.payload)
            break
    return EmailInfo(
        subject=str(msg.get("subject", "") or ""),
        sender=str(msg.get("from", "") or ""),
        to=_addresses(msg.get_all("to", [])),
        cc=_addresses(msg.get_all("cc", [])),
        date=str(msg.get("date", "") or ""),
        message_id=str(msg.get("message-id", "") or "").strip(),
        body=body,
        attachments=tuple(attachments),
        pec=pec,
    )


def _parts(part: Any) -> Iterator[Any]:
    """Depth-first parts, *not* descending into attached messages.

    ``Message.walk`` descends into a ``message/rfc822`` part, so the original
    message inside a PEC gave its body to the envelope's body and its
    attachments to the envelope's attachment list. An attached message is one
    attachment; what it contains is its own.
    """
    yield part
    if part.get_content_type() == "message/rfc822":
        return
    if part.is_multipart():
        for sub in part.iter_parts():
            yield from _parts(sub)


def _payload_bytes(part: Any) -> bytes | None:
    if part.get_content_type() == "message/rfc822":
        inner = part.get_payload()
        if not (isinstance(inner, list) and inner):
            return None
        # RFC 2046 forbids base64 on message/rfc822, and some mailers do it
        # anyway; the "message" parsed from it is then headerless base64 text.
        if str(part.get("content-transfer-encoding", "")).lower() == "base64":
            body = inner[0].get_payload()
            if isinstance(body, str):
                try:
                    return base64.b64decode(re.sub(r"\s+", "", body), validate=True)
                except (binascii.Error, ValueError):
                    pass
        return bytes(inner[0].as_bytes(policy=email.policy.default))
    payload = part.get_payload(decode=True)
    return payload if isinstance(payload, bytes) else None


def _text_of(part: Any) -> str:
    try:
        return str(part.get_content())
    except (LookupError, UnicodeDecodeError):
        payload = part.get_payload(decode=True) or b""
        return payload.decode("utf-8", errors="replace")


def _addresses(values: list[Any]) -> tuple[str, ...]:
    out: list[str] = []
    for v in values:
        out.extend(a.strip() for a in str(v).split(",") if a.strip())
    return tuple(out)


def parse_daticert(raw: bytes) -> dict[str, Any]:
    """The certification data of a PEC: type, sender, recipients, subject,
    timestamp, identifier. Field names follow the ``daticert.xml`` schema."""
    try:
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError:
        return {}
    out: dict[str, Any] = {"tipo": root.get("tipo", ""), "errore": root.get("errore", "")}
    header = root.find("intestazione")
    if header is not None:
        out["mittente"] = (header.findtext("mittente") or "").strip()
        out["destinatari"] = [
            (d.text or "").strip() for d in header.findall("destinatari") if d.text
        ]
        out["oggetto"] = (header.findtext("oggetto") or "").strip()
    data = root.find("dati")
    if data is not None:
        out["gestore"] = (data.findtext("gestore-emittente") or "").strip()
        when = data.find("data")
        if when is not None:
            day = (when.findtext("giorno") or "").strip()
            time = (when.findtext("ora") or "").strip()
            out["data"] = f"{day} {time}".strip()
            out["zona"] = when.get("zona", "")
        out["identificativo"] = (data.findtext("identificativo") or "").strip()
        out["msgid"] = (data.findtext("msgid") or "").strip()
        consegna = data.find("consegna")
        if consegna is not None and consegna.text:
            out["consegna"] = consegna.text.strip()
    return {k: v for k, v in out.items() if v not in ("", [], None)}


# --------------------------------------------------------------------------- #
# zip                                                                          #
# --------------------------------------------------------------------------- #


def zip_members(raw: bytes, *, max_members: int = 10_000) -> Iterator[tuple[str, bytes]]:
    """(name, bytes) for each file in an archive. Encrypted members are skipped
    rather than failing the archive; directories and macOS metadata too."""
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        for n, info in enumerate(zf.infolist()):
            if n >= max_members:
                break
            name = info.filename
            if info.is_dir() or name.startswith("__MACOSX/") or info.flag_bits & 0x1:
                continue
            try:
                yield name, zf.read(info)
            except (RuntimeError, zipfile.BadZipFile, NotImplementedError):
                continue


# --------------------------------------------------------------------------- #
# html                                                                         #
# --------------------------------------------------------------------------- #


def html_to_text(html: str) -> str:
    """Readable text from HTML: tags dropped, blocks separated, entities decoded."""
    from html.parser import HTMLParser

    class _Text(HTMLParser):
        BLOCK: ClassVar[frozenset[str]] = frozenset(
            "p div br li tr h1 h2 h3 h4 h5 h6 table section article blockquote pre".split()  # noqa: SIM905
        )

        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.out: list[str] = []
            self.skip = 0

        def handle_starttag(self, tag: str, attrs: Any) -> None:
            if tag in ("script", "style", "head"):
                self.skip += 1
            elif tag in self.BLOCK:
                self.out.append("\n")

        def handle_endtag(self, tag: str) -> None:
            if tag in ("script", "style", "head"):
                self.skip = max(0, self.skip - 1)
            elif tag in self.BLOCK:
                self.out.append("\n")

        def handle_data(self, data: str) -> None:
            if not self.skip:
                self.out.append(data)

    p = _Text()
    p.feed(html)
    text = "".join(p.out)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
