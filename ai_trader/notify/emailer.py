"""SMTP email sender for daily reports and Merrill order tickets.

Works with any SMTP provider (Gmail app-password, Outlook, SES...).
The recipient is whatever you set in config.yaml -> reports.recipient_email.
If SMTP is not configured, messages are written to reports/outbox/ so
nothing is ever lost.
"""

from __future__ import annotations

import logging
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

logger = logging.getLogger(__name__)


class Emailer:
    def __init__(
        self,
        host: str | None,
        port: int,
        username: str | None,
        password: str | None,
        outbox_dir: str | Path = "reports/outbox",
    ):
        self.host, self.port = host, port
        self.username, self.password = username, password
        self.outbox = Path(outbox_dir)
        self.outbox.mkdir(parents=True, exist_ok=True)

    @property
    def configured(self) -> bool:
        return bool(self.host and self.username and self.password)

    def send(self, to: str, subject: str, body: str, html: str | None = None) -> bool:
        if not self.configured:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = self.outbox / f"{stamp}_{subject[:40].replace(' ', '_').replace('/', '-')}.txt"
            path.write_text(f"To: {to}\nSubject: {subject}\n\n{body}")
            logger.warning("SMTP not configured — message saved to %s", path)
            return False

        msg = MIMEMultipart("alternative")
        msg["From"], msg["To"], msg["Subject"] = self.username, to, subject
        msg.attach(MIMEText(body, "plain"))
        if html:
            msg.attach(MIMEText(html, "html"))

        try:
            with smtplib.SMTP(self.host, self.port, timeout=30) as smtp:
                smtp.starttls()
                smtp.login(self.username, self.password)
                smtp.sendmail(self.username, [to], msg.as_string())
            logger.info("email sent to %s: %s", to, subject)
            return True
        except Exception:
            logger.exception("email send failed; saving to outbox")
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            (self.outbox / f"{stamp}_FAILED.txt").write_text(
                f"To: {to}\nSubject: {subject}\n\n{body}"
            )
            return False
