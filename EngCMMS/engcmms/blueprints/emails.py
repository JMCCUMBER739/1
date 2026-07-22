"""Email templates, composer, digest generator and sent-email log."""

from __future__ import annotations

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import login_required

from .. import analytics, email_service
from ..extensions import db
from ..models import Contact, EmailLog, EmailTemplate
from ..permissions import editor_required

bp = Blueprint("emails", __name__, url_prefix="/emails")


@bp.route("/")
@login_required
def index():
    templates = EmailTemplate.query.order_by(EmailTemplate.category, EmailTemplate.name).all()
    logs = EmailLog.query.order_by(EmailLog.created_at.desc()).limit(50).all()
    return render_template("emails/index.html", templates=templates, logs=logs)


@bp.route("/templates/new", methods=["GET", "POST"])
@bp.route("/templates/<int:template_id>/edit", methods=["GET", "POST"])
@login_required
@editor_required
def edit_template(template_id=None):
    template = db.get_or_404(EmailTemplate, template_id) if template_id else None
    if request.method == "POST":
        f = request.form
        if template is None:
            template = EmailTemplate()
            db.session.add(template)
        elif template.is_builtin:
            # Allow editing the body of built-ins but keep them intact.
            pass
        template.name = f.get("name", "").strip()
        template.category = f.get("category", "general")
        template.subject = f.get("subject", "").strip()
        template.body = f.get("body", "")
        template.description = f.get("description", "")
        if not template.key:
            template.key = (template.name or "template").lower().replace(" ", "_")[:60]
        if not template.name or not template.subject:
            flash("Name and subject are required.", "danger")
        else:
            db.session.commit()
            flash("Template saved.", "success")
            return redirect(url_for("emails.index"))
    return render_template("emails/edit_template.html", template=template)


@bp.route("/compose", methods=["GET", "POST"])
@login_required
@editor_required
def compose():
    templates = EmailTemplate.query.order_by(EmailTemplate.name).all()
    if request.method == "POST":
        subject = request.form.get("subject", "").strip()
        body = request.form.get("body", "")
        category = request.form.get("category", "general")
        recipients_raw = request.form.get("recipients", "").strip()

        if recipients_raw:
            recipients = [r.strip() for r in recipients_raw.replace(";", ",").split(",") if r.strip()]
        else:
            recipients = email_service.recipients_for(category)

        if not subject:
            flash("Subject is required.", "danger")
        else:
            log = email_service.send_email(subject, recipients, body, category=category)
            flash(f"Email {log.status} to {len(recipients)} recipient(s).", "info")
            return redirect(url_for("emails.index"))

    preset = None
    template_id = request.args.get("template", type=int)
    if template_id:
        preset = db.session.get(EmailTemplate, template_id)
    return render_template(
        "emails/compose.html", templates=templates, preset=preset,
        contacts=Contact.query.filter_by(active=True).order_by(Contact.name).all(),
    )


@bp.route("/template/<int:template_id>.json")
@login_required
def template_json(template_id):
    template = db.get_or_404(EmailTemplate, template_id)
    return {
        "subject": template.subject,
        "body": template.body,
        "category": template.category,
    }


@bp.route("/digest", methods=["POST"])
@login_required
@editor_required
def send_digest():
    """Generate and send the weekly digest using live KPIs."""
    template = EmailTemplate.query.filter_by(key="weekly_update").first()
    kpis = analytics.dashboard_kpis()
    ctx = {**email_service.base_context(), **kpis,
           "notes": request.form.get("notes", "No additional notes.")}
    if template:
        subject, body = email_service.render_template_record(template, ctx)
    else:
        subject = "Weekly Engineering & Maintenance Update"
        body = str(kpis)
    recipients = email_service.recipients_for("digest")
    log = email_service.send_email(subject, recipients, body, category="digest")
    flash(f"Weekly digest {log.status} to {len(recipients)} contact(s).", "info")
    return redirect(url_for("emails.index"))


@bp.route("/log/<int:log_id>")
@login_required
def view_log(log_id):
    log = db.get_or_404(EmailLog, log_id)
    return render_template("emails/view_log.html", log=log)
