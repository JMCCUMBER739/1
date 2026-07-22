"""Optional IMAP mail poller for EngCMMS email-to-server intake.

Run this on a schedule (Windows Task Scheduler / cron). It logs into a mailbox,
reads unseen messages, infers the submission type from the subject prefix
(e.g. "IP:", "REQ:", "CDR:"), extracts a project reference, and POSTs each
message (plus its first attachment) to the EngCMMS intake API.

Configuration via environment variables:
    POLL_IMAP_HOST, POLL_IMAP_USER, POLL_IMAP_PASSWORD
    POLL_IMAP_FOLDER          (default: INBOX)
    POLL_ENGCMMS_URL          (e.g. http://engcmms.lab.local:8000)
    POLL_ENGCMMS_TOKEN        (must match ENGCMMS_INTAKE_TOKEN on the server)
    POLL_APPLY                (1 to auto-apply to the design board; default 0)

Requires the 'requests' package (pip install requests).

This is a convenience helper and is intentionally dependency-light; adapt it to
your mail environment and security requirements.
"""

from __future__ import annotations

import email
import imaplib
import os
import re
import sys

TYPE_PREFIXES = ["REQ", "IP", "CDR", "DR", "DWG", "TPR", "AUDIT"]


def infer_type(subject: str) -> str:
    upper = (subject or "").upper()
    for code in TYPE_PREFIXES:
        if re.match(rf"^\s*{code}\b|\b{code}\s*[:\-]", upper):
            return code
    return "OTHER"


def infer_project(subject: str) -> str:
    # Look for a code like DX-101, L-XX-1234-01 or H-XX-1234-01.
    m = re.search(r"\b([A-Z]{1,3}-[A-Z0-9]{1,4}-?\d{1,5}(?:-\d{1,2})?)\b", subject or "")
    return m.group(1) if m else ""


def first_attachment(msg):
    for part in msg.walk():
        if part.get_content_disposition() == "attachment":
            filename = part.get_filename() or "attachment"
            return filename, part.get_payload(decode=True)
    return None, None


def main() -> int:
    try:
        import requests
    except ImportError:
        print("Please 'pip install requests' to use the poller.")
        return 1

    host = os.environ.get("POLL_IMAP_HOST")
    user = os.environ.get("POLL_IMAP_USER")
    password = os.environ.get("POLL_IMAP_PASSWORD")
    base_url = os.environ.get("POLL_ENGCMMS_URL", "http://127.0.0.1:5000").rstrip("/")
    token = os.environ.get("POLL_ENGCMMS_TOKEN", "")
    folder = os.environ.get("POLL_IMAP_FOLDER", "INBOX")
    apply_flag = "1" if os.environ.get("POLL_APPLY", "0") in ("1", "true", "yes") else "0"

    if not (host and user and password and token):
        print("Missing POLL_IMAP_HOST/USER/PASSWORD or POLL_ENGCMMS_TOKEN.")
        return 1

    imap = imaplib.IMAP4_SSL(host)
    imap.login(user, password)
    imap.select(folder)
    _typ, data = imap.search(None, "UNSEEN")
    ids = data[0].split()
    print(f"Found {len(ids)} unseen message(s).")

    for num in ids:
        _typ, msg_data = imap.fetch(num, "(RFC822)")
        msg = email.message_from_bytes(msg_data[0][1])
        subject = str(email.header.make_header(email.header.decode_header(msg.get("Subject", ""))))
        sender = email.utils.parseaddr(msg.get("From", ""))[1]
        body = ""
        for part in msg.walk():
            if part.get_content_type() == "text/plain" and part.get_content_disposition() is None:
                body = part.get_payload(decode=True).decode(errors="replace")
                break

        fname, fbytes = first_attachment(msg)
        payload = {
            "type": infer_type(subject),
            "project": infer_project(subject),
            "from": sender,
            "subject": subject,
            "body": body,
            "apply": apply_flag,
        }
        files = {"file": (fname, fbytes)} if fbytes else None
        try:
            resp = requests.post(
                f"{base_url}/intake/api",
                data=payload,
                files=files,
                headers={"X-Intake-Token": token},
                timeout=30,
            )
            print(f"  {subject!r} -> {resp.status_code} {resp.text[:120]}")
        except Exception as exc:  # pragma: no cover
            print(f"  POST failed for {subject!r}: {exc}")

    imap.logout()
    return 0


if __name__ == "__main__":
    sys.exit(main())
