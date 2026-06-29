"""Email auto-generation, templating and (optional) SMTP delivery.

When no SMTP server is configured the service runs in *preview* mode: messages
are fully rendered and written to the :class:`EmailLog` so they can be reviewed
in the UI, but nothing leaves the machine.  This makes local office testing safe
and lets the operator wire up the internal mail relay only when ready.
"""

from __future__ import annotations

import re
import smtplib
from email.message import EmailMessage

from flask import current_app

from .extensions import db
from .models import Contact, EmailLog, EmailTemplate

_PLACEHOLDER = re.compile(r"{{\s*([\w_]+)\s*}}")


def generate_email_address(full_name: str, domain: str | None = None) -> str:
    """Turn a person's name into ``first.last@domain`` (lowercase, ascii-safe)."""
    domain = domain or current_app.config.get("EMAIL_DOMAIN", "lab.local")
    parts = re.sub(r"[^a-zA-Z\s]", "", full_name).strip().lower().split()
    if not parts:
        return f"user@{domain}"
    if len(parts) == 1:
        local = parts[0]
    else:
        local = f"{parts[0]}.{parts[-1]}"
    return f"{local}@{domain}"


def render(text: str, context: dict) -> str:
    """Replace ``{{ placeholder }}`` tokens with values from *context*."""
    def _sub(match: re.Match) -> str:
        key = match.group(1)
        value = context.get(key, match.group(0))
        return "" if value is None else str(value)

    return _PLACEHOLDER.sub(_sub, text or "")


def render_template_record(template: EmailTemplate, context: dict) -> tuple[str, str]:
    return render(template.subject, context), render(template.body, context)


def recipients_for(category: str) -> list[str]:
    """Return active contact emails subscribed to a notification *category*."""
    field = {
        "workorder": Contact.notify_workorders,
        "pm": Contact.notify_pm,
        "calibration": Contact.notify_calibration,
        "project": Contact.notify_projects,
        "digest": Contact.notify_digest,
    }.get(category)

    query = Contact.query.filter_by(active=True)
    if field is not None:
        query = query.filter(field.is_(True))
    return [c.email for c in query.all() if c.email]


def send_email(subject: str, recipients, body: str, category: str = "general") -> EmailLog:
    """Send (or preview) an email and record it in the log."""
    if isinstance(recipients, str):
        recipients = [recipients]
    recipients = [r for r in recipients if r]

    log = EmailLog(
        subject=subject,
        recipients=", ".join(recipients),
        body=body,
        category=category,
    )

    server = current_app.config.get("MAIL_SERVER")
    if not server or not recipients:
        log.status = "preview"
        db.session.add(log)
        db.session.commit()
        return log

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = current_app.config["MAIL_DEFAULT_SENDER"]
        msg["To"] = ", ".join(recipients)
        msg.set_content(body)

        port = current_app.config.get("MAIL_PORT", 25)
        with smtplib.SMTP(server, port, timeout=20) as smtp:
            if current_app.config.get("MAIL_USE_TLS"):
                smtp.starttls()
            username = current_app.config.get("MAIL_USERNAME")
            if username:
                smtp.login(username, current_app.config.get("MAIL_PASSWORD", ""))
            smtp.send_message(msg)
        log.status = "sent"
    except Exception as exc:  # pragma: no cover - depends on network
        log.status = "failed"
        log.error = str(exc)

    db.session.add(log)
    db.session.commit()
    return log


def base_context() -> dict:
    return {"org_name": current_app.config.get("ORG_NAME", "Engineering")}


# ---------------------------------------------------------------------------
# Auto-generated notifications triggered by CMMS events
# ---------------------------------------------------------------------------
def _template(key: str) -> EmailTemplate | None:
    return EmailTemplate.query.filter_by(key=key).first()


def notify_workorder_event(workorder, template_key: str = "wo_assigned") -> EmailLog | None:
    template = _template(template_key)
    recipients = recipients_for("workorder")
    if workorder.assignee and workorder.assignee.email:
        recipients = list({*recipients, workorder.assignee.email})
    if not recipients:
        return None

    ctx = {
        **base_context(),
        "wo_number": workorder.number,
        "wo_title": workorder.title,
        "wo_priority": workorder.priority,
        "wo_status": workorder.status,
        "wo_type": workorder.wo_type,
        "asset_name": workorder.asset.name if workorder.asset else "N/A",
        "assignee": workorder.assignee.full_name if workorder.assignee else "Unassigned",
        "due_date": workorder.due_date.strftime("%Y-%m-%d") if workorder.due_date else "Not set",
        "description": workorder.description or "",
    }
    if template:
        subject, body = render_template_record(template, ctx)
    else:
        subject = f"[{ctx['wo_priority'].upper()}] Work Order {ctx['wo_number']}: {ctx['wo_title']}"
        body = (
            f"Work order {ctx['wo_number']} ({ctx['wo_type']}) on {ctx['asset_name']} "
            f"is now '{ctx['wo_status']}'.\nAssignee: {ctx['assignee']}\n"
            f"Due: {ctx['due_date']}\n\n{ctx['description']}"
        )
    return send_email(subject, recipients, body, category="workorder")


def notify_request_received(request_obj) -> EmailLog | None:
    recipients = recipients_for("workorder")
    if not recipients:
        return None
    subject = f"New maintenance request: {request_obj.subject}"
    body = (
        f"A new maintenance request was logged via {request_obj.source}.\n\n"
        f"From: {request_obj.requester_name or 'Unknown'} "
        f"<{request_obj.requester_email or 'n/a'}>\n"
        f"Priority: {request_obj.priority}\n\n{request_obj.body or ''}"
    )
    return send_email(subject, recipients, body, category="workorder")
