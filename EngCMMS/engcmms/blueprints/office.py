"""Team template library: upload Word / ODF / PDF templates and email them out."""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import date

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename

from .. import email_service
from ..extensions import db
from ..models import TEMPLATE_EXTENSIONS, Contact, MessageTemplate
from ..permissions import editor_required

bp = Blueprint("office", __name__, url_prefix="/team-templates")


def _allowed(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in TEMPLATE_EXTENSIONS


@bp.route("/")
@login_required
def list_templates():
    templates = MessageTemplate.query.order_by(MessageTemplate.category, MessageTemplate.name).all()
    return render_template("office/list.html", templates=templates)


@bp.route("/upload", methods=["GET", "POST"])
@login_required
@editor_required
def upload():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename:
            flash("Please choose a file.", "danger")
            return redirect(url_for("office.upload"))
        if not _allowed(file.filename):
            flash("Allowed types: Word, ODF, PDF, RTF, TXT, XLSX, PPTX.", "danger")
            return redirect(url_for("office.upload"))

        original = secure_filename(file.filename)
        stored = f"tpl_{uuid.uuid4().hex}_{original}"
        path = os.path.join(current_app.config["UPLOAD_DIR"], stored)
        file.save(path)

        tpl = MessageTemplate(
            name=request.form.get("name", "").strip() or original,
            description=request.form.get("description", ""),
            category=request.form.get("category", "general"),
            stored_filename=stored,
            original_filename=original,
            content_type=file.mimetype,
            size_bytes=os.path.getsize(path),
            uploaded_by_id=current_user.id,
        )
        db.session.add(tpl)
        db.session.commit()
        flash("Template uploaded.", "success")
        return redirect(url_for("office.list_templates"))

    return render_template("office/upload.html")


@bp.route("/<int:tpl_id>/download")
@login_required
def download(tpl_id):
    tpl = db.get_or_404(MessageTemplate, tpl_id)
    directory = current_app.config["UPLOAD_DIR"]
    if not os.path.exists(os.path.join(directory, tpl.stored_filename)):
        abort(404)
    return send_from_directory(directory, tpl.stored_filename, as_attachment=True,
                               download_name=tpl.original_filename)


@bp.route("/<int:tpl_id>/delete", methods=["POST"])
@login_required
@editor_required
def delete(tpl_id):
    tpl = db.get_or_404(MessageTemplate, tpl_id)
    path = os.path.join(current_app.config["UPLOAD_DIR"], tpl.stored_filename)
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
    db.session.delete(tpl)
    db.session.commit()
    flash("Template deleted.", "info")
    return redirect(url_for("office.list_templates"))


@bp.route("/<int:tpl_id>/send", methods=["GET", "POST"])
@login_required
@editor_required
def send(tpl_id):
    tpl = db.get_or_404(MessageTemplate, tpl_id)
    src_path = os.path.join(current_app.config["UPLOAD_DIR"], tpl.stored_filename)

    if request.method == "POST":
        subject = request.form.get("subject", "").strip() or tpl.name
        message = request.form.get("message", "")
        category = request.form.get("category", "general")
        recipients_raw = request.form.get("recipients", "").strip()
        if recipients_raw:
            recipients = [r.strip() for r in recipients_raw.replace(";", ",").split(",") if r.strip()]
        else:
            recipients = email_service.recipients_for(category)

        # Optionally fill {{ }} placeholders in .docx templates.
        attach_path, attach_name = src_path, tpl.original_filename
        temp_path = None
        if tpl.extension == "docx" and request.form.get("fill") == "on":
            ctx = {**email_service.base_context(), "date": date.today().isoformat()}
            for pair in request.form.get("tokens", "").splitlines():
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    ctx[k.strip()] = v.strip()
            fd, temp_path = tempfile.mkstemp(suffix=".docx")
            os.close(fd)
            if email_service.fill_docx_placeholders(src_path, temp_path, ctx):
                attach_path, attach_name = temp_path, tpl.original_filename
            else:
                flash("python-docx not installed — sent the template unmodified.", "warning")

        log = email_service.send_email(subject, recipients, message, category=category,
                                       attachments=[(attach_path, attach_name)])
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        flash(f"Template '{tpl.name}' email {log.status} to {len(recipients)} recipient(s).", "info")
        return redirect(url_for("office.list_templates"))

    return render_template(
        "office/send.html", tpl=tpl,
        contacts=Contact.query.filter_by(active=True).order_by(Contact.name).all(),
    )
