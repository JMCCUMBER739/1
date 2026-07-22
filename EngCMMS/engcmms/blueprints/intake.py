"""Email-to-server intake.

Teams submit design documents (Requirements, Implementation Plan, CDR, DR,
Drawings, Test Plan Results, Audit) to the server. Each submission can be:

* logged manually via the form, or
* posted by a mail gateway / IMAP poller to the token-protected API endpoint.

Applying a submission advances the matching maturity stage on its linked
design project and files the attached document.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime

from flask import (
    Blueprint,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename

from ..extensions import db
from ..models import (
    SUBMISSION_TYPE_FIELD,
    SUBMISSION_TYPE_LABEL,
    SUBMISSION_TYPES,
    DesignProject,
    Document,
    Submission,
)
from ..permissions import editor_required

bp = Blueprint("intake", __name__, url_prefix="/intake")


def _save_document(file, title, category):
    """Persist an uploaded file as a Document and return it (or None)."""
    if not file or not file.filename:
        return None
    original = secure_filename(file.filename)
    stored = f"sub_{uuid.uuid4().hex}_{original}"
    path = os.path.join(current_app.config["UPLOAD_DIR"], stored)
    file.save(path)
    doc = Document(
        title=title or original,
        category=category,
        stored_filename=stored,
        original_filename=original,
        content_type=file.mimetype,
        size_bytes=os.path.getsize(path),
    )
    db.session.add(doc)
    db.session.flush()
    return doc


def _match_project(ref):
    """Best-effort match of a free-text reference to a design project."""
    if not ref:
        return None
    ref = ref.strip()
    if ref.isdigit():
        dp = DesignProject.query.filter_by(number=int(ref)).first()
        if dp:
            return dp
    like = f"%{ref}%"
    return DesignProject.query.filter(db.or_(
        DesignProject.windchill_number.ilike(like),
        DesignProject.epdm_number.ilike(like),
        DesignProject.title.ilike(like),
    )).first()


def _apply_submission(sub: Submission) -> bool:
    """Advance the matching design stage. Returns True if applied."""
    field = SUBMISSION_TYPE_FIELD.get(sub.submission_type)
    if not sub.design_project or not field:
        return False
    current = getattr(sub.design_project, field) or 0
    target = sub.resulting_code if sub.resulting_code is not None else 2
    # Never downgrade a released/revised stage.
    setattr(sub.design_project, field, max(current, target) if current in (3, 4) else target)
    sub.status = "applied"
    sub.applied_at = datetime.utcnow()
    return True


@bp.route("/")
@login_required
def inbox():
    status = request.args.get("status", "")
    query = Submission.query
    if status:
        query = query.filter_by(status=status)
    items = query.order_by(Submission.created_at.desc()).all()
    return render_template(
        "intake/inbox.html", items=items, status=status,
        types=SUBMISSION_TYPES, type_labels=SUBMISSION_TYPE_LABEL,
        projects=DesignProject.query.order_by(DesignProject.number.asc().nullslast()).all(),
    )


@bp.route("/new", methods=["POST"])
@login_required
@editor_required
def create():
    f = request.form
    stype = f.get("submission_type", "OTHER")
    dp_id = int(f["design_project_id"]) if f.get("design_project_id") else None
    dp = db.session.get(DesignProject, dp_id) if dp_id else _match_project(f.get("project_ref"))

    doc = _save_document(request.files.get("file"),
                         title=f.get("subject") or SUBMISSION_TYPE_LABEL.get(stype),
                         category="report")
    sub = Submission(
        submission_type=stype,
        design_project_id=dp.id if dp else None,
        project_ref=f.get("project_ref", ""),
        sender_name=f.get("sender_name", ""),
        sender_email=f.get("sender_email", ""),
        subject=f.get("subject", ""),
        body=f.get("body", ""),
        document_id=doc.id if doc else None,
        resulting_code=int(f.get("resulting_code") or 2),
        source="email" if f.get("source") == "email" else "manual",
        status="received",
    )
    db.session.add(sub)
    db.session.flush()  # make relationships available before applying
    if f.get("apply_now") == "on":
        _apply_submission(sub)
    db.session.commit()
    flash("Submission logged." + (" Stage updated." if sub.status == "applied" else ""), "success")
    return redirect(url_for("intake.inbox"))


@bp.route("/<int:sub_id>/apply", methods=["POST"])
@login_required
@editor_required
def apply(sub_id):
    sub = db.get_or_404(Submission, sub_id)
    if _apply_submission(sub):
        db.session.commit()
        flash("Submission applied to the design board.", "success")
    else:
        flash("Could not apply — link a design project and a mappable type first.", "warning")
    return redirect(url_for("intake.inbox"))


@bp.route("/<int:sub_id>/reject", methods=["POST"])
@login_required
@editor_required
def reject(sub_id):
    sub = db.get_or_404(Submission, sub_id)
    sub.status = "rejected"
    db.session.commit()
    flash("Submission rejected.", "info")
    return redirect(url_for("intake.inbox"))


@bp.route("/api", methods=["POST"])
def api():
    """Machine-to-machine intake for a mail gateway / IMAP poller.

    Auth: send the shared token in the ``X-Intake-Token`` header or a ``token``
    field. Disabled unless ``ENGCMMS_INTAKE_TOKEN`` is configured.

    Accepts form fields: type, project, from, from_name, subject, body,
    resulting_code, apply (1/0) and an optional ``file`` attachment.
    """
    expected = current_app.config.get("INTAKE_TOKEN")
    if not expected:
        return jsonify({"ok": False, "error": "intake API disabled (no token configured)"}), 403
    provided = request.headers.get("X-Intake-Token") or request.form.get("token")
    if provided != expected:
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    f = request.form
    stype = (f.get("type") or "OTHER").upper()
    if stype not in SUBMISSION_TYPE_FIELD:
        stype = "OTHER"
    dp = _match_project(f.get("project"))
    doc = _save_document(request.files.get("file"),
                         title=f.get("subject") or SUBMISSION_TYPE_LABEL.get(stype),
                         category="report")
    sub = Submission(
        submission_type=stype,
        design_project_id=dp.id if dp else None,
        project_ref=f.get("project", ""),
        sender_name=f.get("from_name", ""),
        sender_email=f.get("from", ""),
        subject=f.get("subject", ""),
        body=f.get("body", ""),
        document_id=doc.id if doc else None,
        resulting_code=int(f.get("resulting_code") or 2),
        source="api",
        status="received",
    )
    db.session.add(sub)
    db.session.flush()  # make relationships available before applying
    applied = False
    if f.get("apply") in ("1", "true", "on", "yes"):
        applied = _apply_submission(sub)
    db.session.commit()
    return jsonify({
        "ok": True, "submission_id": sub.id, "matched_project": bool(dp),
        "applied": applied, "type": stype,
    })
