"""The filesystem scanner on a company archive: containers, sidecars, ACLs, reuse.

Fixtures are built here rather than checked in -- a signed envelope, an email
with attachments, a PEC, a zip -- so each test shows exactly the structure it
relies on.
"""

from __future__ import annotations

import datetime
import io
import json
import zipfile
from email.message import EmailMessage
from pathlib import Path

import pytest

from indexer.impls.containers import parse_daticert, parse_email, unwrap_p7m
from indexer.impls.corpus import FilesystemScanner
from indexer.pipeline import assemble

DATICERT = b"""<?xml version="1.0" encoding="UTF-8"?>
<postacert tipo="posta-certificata" errore="nessuno">
  <intestazione>
    <mittente>mario.rossi@pec.it</mittente>
    <destinatari tipo="certificato">ufficio@pec.azienda.it</destinatari>
    <oggetto>Invio contratto</oggetto>
  </intestazione>
  <dati>
    <gestore-emittente>Gestore PEC S.p.A.</gestore-emittente>
    <data zona="+0100"><giorno>15/01/2025</giorno><ora>10:30:00</ora></data>
    <identificativo>opec123.20250115103000.01@pec.gestore.it</identificativo>
    <msgid>&lt;abc@example.it&gt;</msgid>
  </dati>
</postacert>"""


def _signed(content: bytes, signer: str = "Mario Rossi") -> bytes:
    crypto = pytest.importorskip("cryptography")
    assert crypto
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives.serialization import pkcs7
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, signer),
            x509.NameAttribute(NameOID.SERIAL_NUMBER, "TINIT-RSSMRA80A01H501U"),
        ]
    )
    now = datetime.datetime(2025, 1, 1, tzinfo=datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=365))
        .sign(key, hashes.SHA256())
    )
    return (
        pkcs7.PKCS7SignatureBuilder()
        .set_data(content)
        .add_signer(cert, key, hashes.SHA256())
        .sign(serialization.Encoding.DER, [pkcs7.PKCS7Options.Binary])
    )


def _email(subject: str, body: str, attachments: list[tuple[str, bytes, str]]) -> bytes:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = "Anna Bianchi <anna@azienda.it>"
    msg["To"] = "ufficio@azienda.it"
    msg["Date"] = "Wed, 15 Jan 2025 10:30:00 +0100"
    msg.set_content(body)
    for name, payload, ctype in attachments:
        main, sub = ctype.split("/")
        msg.add_attachment(payload, maintype=main, subtype=sub, filename=name)
    return bytes(msg)


def _docs(scanner: FilesystemScanner) -> dict[str, object]:
    return {d.metadata["relpath"]: d for d in scanner.scan()}


class TestStatCache:
    def test_unchanged_files_are_not_read_again(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("alpha")
        (tmp_path / "b.txt").write_text("beta")
        s = FilesystemScanner({"root": str(tmp_path)}, state_dir=tmp_path / ".state")
        first = _docs(s)
        assert s.files_read == 2
        again = FilesystemScanner({"root": str(tmp_path)}, state_dir=tmp_path / ".state")
        second = _docs(again)
        assert again.files_read == 0
        assert {k: d.content_hash for k, d in first.items()} == {  # type: ignore[attr-defined]
            k: d.content_hash  # type: ignore[attr-defined]
            for k, d in second.items()
        }
        (tmp_path / "b.txt").write_text("beta, edited")
        third = FilesystemScanner({"root": str(tmp_path)}, state_dir=tmp_path / ".state")
        _docs(third)
        assert third.files_read == 1


class TestSidecarsAndAcls:
    def test_sidecar_supplies_id_acl_and_facts(self, tmp_path: Path) -> None:
        (tmp_path / "contratto.txt").write_text("Contratto di fornitura")
        (tmp_path / "contratto.txt.meta.json").write_text(
            json.dumps({"document_id": "DMS-0042", "acl": ["group:legal"], "cliente": "C001"})
        )
        docs = _docs(FilesystemScanner({"root": str(tmp_path)}))
        assert list(docs) == ["contratto.txt"]  # the sidecar is not a document
        d = docs["contratto.txt"]
        assert d.metadata["acl"] == ["group:legal"]  # type: ignore[attr-defined]
        assert d.metadata["cliente"] == "C001"  # type: ignore[attr-defined]
        assert d.metadata["source_id"] == "DMS-0042"  # type: ignore[attr-defined]
        # The source system's id survives a move; the path-derived one would not.
        (tmp_path / "archivio").mkdir()
        (tmp_path / "contratto.txt").rename(tmp_path / "archivio" / "contratto.txt")
        (tmp_path / "contratto.txt.meta.json").rename(
            tmp_path / "archivio" / "contratto.txt.meta.json"
        )
        moved = _docs(FilesystemScanner({"root": str(tmp_path)}))
        assert moved["archivio/contratto.txt"].document_id == d.document_id  # type: ignore[attr-defined]

    def test_a_malformed_sidecar_withholds_rather_than_publishes(self, tmp_path: Path) -> None:
        (tmp_path / "x.txt").write_text("riservato")
        (tmp_path / "x.txt.meta.json").write_text("{not json")
        d = _docs(FilesystemScanner({"root": str(tmp_path), "default_acl": ["public"]}))["x.txt"]
        assert d.metadata["acl"] == []  # type: ignore[attr-defined]
        assert "sidecar_error" in d.metadata  # type: ignore[attr-defined]

    def test_acl_rules_by_folder_and_default(self, tmp_path: Path) -> None:
        (tmp_path / "hr").mkdir()
        (tmp_path / "hr" / "cedolino.txt").write_text("stipendio")
        (tmp_path / "circolare.txt").write_text("orari")
        s = FilesystemScanner(
            {
                "root": str(tmp_path),
                "acl_rules": [{"pattern": "hr/**", "acl": ["group:hr"]}],
                "default_acl": ["group:all"],
            }
        )
        docs = _docs(s)
        assert docs["hr/cedolino.txt"].metadata["acl"] == ["group:hr"]  # type: ignore[attr-defined]
        assert docs["circolare.txt"].metadata["acl"] == ["group:all"]  # type: ignore[attr-defined]


class TestSignedEnvelopes:
    def test_p7m_yields_the_signed_document(self, tmp_path: Path) -> None:
        inner = b"Contratto firmato digitalmente tra le parti."
        (tmp_path / "contratto.txt.p7m").write_bytes(_signed(inner))
        docs = _docs(FilesystemScanner({"root": str(tmp_path), "expand": ["p7m"]}))
        d = docs["contratto.txt.p7m"]
        assert d.metadata["name"] == "contratto.txt"  # type: ignore[attr-defined]
        assert d.media_type == "text/plain"  # type: ignore[attr-defined]
        assert d.load() == inner  # type: ignore[attr-defined]
        assert d.metadata["signers"] == ["Mario Rossi"]  # type: ignore[attr-defined]
        assert d.metadata["signer_ids"] == ["TINIT-RSSMRA80A01H501U"]  # type: ignore[attr-defined]
        assert d.metadata["signature_verified"] is False  # type: ignore[attr-defined]

    def test_base64_and_nested_envelopes(self) -> None:
        import base64

        inner = b"<FatturaElettronica/>"
        once = _signed(inner)
        assert unwrap_p7m(base64.b64encode(once)).content == inner
        twice = unwrap_p7m(_signed(once))
        assert (twice.content, twice.layers) == (inner, 2)


class TestEmail:
    def test_attachments_become_child_documents(self, tmp_path: Path) -> None:
        raw = _email(
            "Ordine 2025/17",
            "In allegato l'ordine e la fattura firmata.",
            [
                ("ordine.txt", b"Ordine di 10 pezzi", "text/plain"),
                ("fattura.xml.p7m", _signed(b"<Fattura/>"), "application/pkcs7-mime"),
            ],
        )
        (tmp_path / "mail.eml").write_bytes(raw)
        docs = _docs(FilesystemScanner({"root": str(tmp_path), "expand": ["eml", "p7m"]}))
        assert set(docs) == {
            "mail.eml",
            "mail.eml#att/0-ordine.txt",
            "mail.eml#att/1-fattura.xml.p7m",
        }
        mail = docs["mail.eml"]
        order = docs["mail.eml#att/0-ordine.txt"]
        invoice = docs["mail.eml#att/1-fattura.xml.p7m"]
        assert mail.metadata["email_subject"] == "Ordine 2025/17"  # type: ignore[attr-defined]
        assert order.metadata["parent_document_id"] == mail.document_id  # type: ignore[attr-defined]
        assert order.metadata["email_subject"] == "Ordine 2025/17"  # type: ignore[attr-defined]
        assert order.load() == b"Ordine di 10 pezzi"  # type: ignore[attr-defined]
        assert invoice.load() == b"<Fattura/>"  # type: ignore[attr-defined]
        assert invoice.media_type == "application/xml"  # type: ignore[attr-defined]
        assert invoice.metadata["signers"] == ["Mario Rossi"]  # type: ignore[attr-defined]

    def test_a_pec_yields_the_original_message_and_its_attachments(self, tmp_path: Path) -> None:
        original = _email(
            "Invio contratto",
            "Buongiorno, invio il contratto.",
            [("contratto.txt", b"Testo del contratto", "text/plain")],
        )
        pec = _email(
            "POSTA CERTIFICATA: Invio contratto",
            "Messaggio di posta certificata: il 15/01/2025 alle 10:30 il messaggio "
            '"Invio contratto" e\' stato inviato da mario.rossi@pec.it',
            [
                ("daticert.xml", DATICERT, "application/xml"),
                ("postacert.eml", original, "message/rfc822"),
                ("smime.p7s", b"signature", "application/pkcs7-signature"),
            ],
        )
        info = parse_email(pec)
        assert info.is_pec and info.pec is not None
        assert info.pec["mittente"] == "mario.rossi@pec.it"
        assert info.pec["tipo"] == "posta-certificata"

        (tmp_path / "pec.eml").write_bytes(pec)
        docs = _docs(FilesystemScanner({"root": str(tmp_path), "expand": ["eml"]}))
        names = sorted(d.metadata["name"] for d in docs.values())  # type: ignore[attr-defined]
        # The envelope's certification data and signature are metadata, not
        # documents; the original message and its attachment are documents --
        # once each: the attachment belongs to the original message, not also
        # to the envelope that carries it.
        assert names == ["contratto.txt", "pec.eml", "postacert.eml"]
        contract = next(d for d in docs.values() if d.metadata["name"] == "contratto.txt")  # type: ignore[attr-defined]
        assert contract.load() == b"Testo del contratto"  # type: ignore[attr-defined]
        assert contract.metadata["pec_identificativo"].startswith("opec123")  # type: ignore[attr-defined]
        assert contract.metadata["email_subject"] == "Invio contratto"  # type: ignore[attr-defined]


class TestZip:
    def test_members_and_nested_containers(self, tmp_path: Path) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("2024/relazione.txt", "Relazione annuale")
            zf.writestr("mail/avviso.eml", _email("Avviso", "Testo", []))
            zf.writestr("__MACOSX/._junk", b"x")
        (tmp_path / "archivio.zip").write_bytes(buf.getvalue())
        docs = _docs(FilesystemScanner({"root": str(tmp_path), "expand": ["zip", "eml"]}))
        assert set(docs) == {"archivio.zip#2024/relazione.txt", "archivio.zip#mail/avviso.eml"}
        report = docs["archivio.zip#2024/relazione.txt"]
        assert report.load() == b"Relazione annuale"  # type: ignore[attr-defined]
        assert report.metadata["archive_path"] == "2024/relazione.txt"  # type: ignore[attr-defined]


class TestDaticert:
    def test_malformed_certification_data_is_empty_not_an_error(self) -> None:
        assert parse_daticert(b"<not xml") == {}


class TestMetadataChangesReachTheIndex:
    def test_an_acl_change_in_a_sidecar_rebuilds_the_document(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        data.mkdir()
        (data / "a.md").write_text("# Nota\n\nContenuto riservato.\n")
        (data / "a.md.meta.json").write_text(json.dumps({"acl": ["group:hr"]}))
        cfg = tmp_path / "c.yaml"
        cfg.write_text(
            f"""
schema_version: 1
project: {{name: s}}
paths: {{store: {tmp_path.as_posix()}/index, cache: {tmp_path.as_posix()}/cache}}
corpus:
  sources:
    - impl: filesystem
      params: {{root: {data.as_posix()}, include: ["**/*.md"]}}
ingestion:
  parse: {{enabled: true, default: {{impl: markdown}}}}
  segment: {{impl: structural}}
  index:
    indexes:
      - {{name: lexical, kind: lexical, impl: bm25_memory}}
query:
  route:
    enabled: true
    impl: rules
    paths:
      structured: {{targets: []}}
      lookup: {{targets: [lexical], step_budget: 1}}
      iterative: {{targets: [lexical], step_budget: 3}}
"""
        )
        assert assemble(cfg).ingestion().build().ok
        (data / "a.md.meta.json").write_text(json.dumps({"acl": ["group:all"]}))
        a = assemble(cfg)
        res = a.ingestion().build()
        # Same bytes, new ACL: previously "unchanged", and the old ACL stayed
        # in every index.
        assert [(c.kind.value, c.reason) for c in res.plan] == [("changed", "metadata changed")]
        stored = [a.unit_store.get(u) for u in a.unit_store.all_ids()]
        assert all(u is not None and u.filter_fields()["acl"] == ("group:all",) for u in stored)
