"""Design-project configuration-management progress board.

Provides an editable, color-coded matrix modeled on the FY design-project
progress-metrics spreadsheet, with inline cell editing, CSV import/export and
email-notification tracking.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime

from flask import (
    Blueprint,
    Response,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user, login_required

from .. import email_service
from ..extensions import db
from ..models import (
    DESIGN_PROJECT_STATUSES,
    DESIGN_STAGE_CODES,
    DESIGN_STAGES,
    DESIGN_TYPES,
    DOC_LOCATIONS,
    DesignProject,
)
from ..permissions import editor_required

bp = Blueprint("design", __name__, url_prefix="/design")

_INT_FIELDS = DesignProject.STAGE_FIELDS
_TEXT_FIELDS = [
    "title", "design_name", "windchill_number", "epdm_number", "program",
    "doc_location", "design_authority", "pm", "dm", "dtl", "da_po", "status",
    "accepted", "accepted_by", "notes",
]


def _coerce_int(value, default=0):
    try:
        return max(0, min(5, int(value)))
    except (TypeError, ValueError):
        return default


@bp.route("/")
@login_required
def board():
    status = request.args.get("status", "")
    program = request.args.get("program", "")
    q = request.args.get("q", "").strip()

    query = DesignProject.query
    if status:
        query = query.filter_by(status=status)
    if program:
        query = query.filter_by(program=program)
    if q:
        like = f"%{q}%"
        query = query.filter(db.or_(
            DesignProject.title.ilike(like),
            DesignProject.design_name.ilike(like),
            DesignProject.windchill_number.ilike(like),
        ))
    projects = query.order_by(DesignProject.number.asc().nullslast(),
                              DesignProject.id.asc()).all()

    programs = sorted({p.program for p in DesignProject.query.all() if p.program})
    from .. import analytics

    return render_template(
        "design/board.html",
        projects=projects,
        metrics=analytics.design_metrics(),
        stages=DESIGN_STAGES,
        stage_codes=DESIGN_STAGE_CODES,
        statuses=DESIGN_PROJECT_STATUSES,
        doc_locations=DOC_LOCATIONS,
        programs=programs,
        status=status, program=program, q=q,
    )


@bp.route("/new", methods=["GET", "POST"])
@bp.route("/<int:dp_id>/edit", methods=["GET", "POST"])
@login_required
@editor_required
def edit(dp_id=None):
    dp = db.get_or_404(DesignProject, dp_id) if dp_id else None
    if request.method == "POST":
        f = request.form
        if dp is None:
            dp = DesignProject()
            db.session.add(dp)
        dp.number = int(f["number"]) if f.get("number") else None
        dp.title = f.get("title", "").strip()
        for field in _TEXT_FIELDS:
            if field == "title":
                continue
            setattr(dp, field, (f.get(field, "") or "").strip() or None)
        if not dp.status:
            dp.status = "A"
        for field in _INT_FIELDS:
            setattr(dp, field, _coerce_int(f.get(field)))
        if not dp.title:
            flash("Title is required.", "danger")
        else:
            db.session.commit()
            flash("Design project saved.", "success")
            return redirect(url_for("design.board"))

    return render_template(
        "design/edit.html", dp=dp, stages=DESIGN_STAGES,
        stage_codes=DESIGN_STAGE_CODES, statuses=DESIGN_PROJECT_STATUSES,
        doc_locations=DOC_LOCATIONS, types=DESIGN_TYPES,
    )


@bp.route("/<int:dp_id>/cell", methods=["POST"])
@login_required
@editor_required
def update_cell(dp_id):
    """Inline update of a single field from the board (AJAX)."""
    dp = db.get_or_404(DesignProject, dp_id)
    field = request.form.get("field", "")
    value = request.form.get("value", "")

    if field in _INT_FIELDS:
        setattr(dp, field, _coerce_int(value))
    elif field in ("status", "accepted", "da_po", "doc_location"):
        setattr(dp, field, value)
    else:
        abort(400)
    db.session.commit()

    new_value = getattr(dp, field)
    return jsonify({
        "ok": True,
        "field": field,
        "value": new_value,
        "maturity": dp.maturity_pct,
        "code_class": f"code-{new_value}" if field in _INT_FIELDS else "",
    })


@bp.route("/<int:dp_id>/delete", methods=["POST"])
@login_required
@editor_required
def delete(dp_id):
    dp = db.get_or_404(DesignProject, dp_id)
    db.session.delete(dp)
    db.session.commit()
    flash("Design project removed.", "info")
    return redirect(url_for("design.board"))


@bp.route("/<int:dp_id>/email", methods=["POST"])
@login_required
@editor_required
def email_update(dp_id):
    dp = db.get_or_404(DesignProject, dp_id)
    recipients = email_service.recipients_for("project")
    subject = f"Design progress update: {dp.title} ({dp.maturity_pct:.0f}% mature)"
    lines = [f"Design project: {dp.title}"]
    if dp.design_name:
        lines.append(f"Design name: {dp.design_name}")
    lines.append(f"WindChill #: {dp.windchill_number or '—'}   ePDM #: {dp.epdm_number or '—'}")
    lines.append(f"PM: {dp.pm or '—'}   DM: {dp.dm or '—'}   DTL: {dp.dtl or '—'}")
    lines.append("")
    for field, abbr, name in DESIGN_STAGES:
        code = getattr(dp, field)
        lines.append(f"  {abbr:6} {name:26} {code} ({DESIGN_STAGE_CODES.get(code, '')})")
    lines.append("")
    lines.append(f"Overall maturity: {dp.maturity_pct:.0f}%")
    lines.append(f"Acceptance: {dp.accept_display}")
    body = "\n".join(lines)

    log = email_service.send_email(subject, recipients, body, category="project")
    dp.email_sent = True
    dp.email_sent_at = datetime.utcnow()
    db.session.commit()
    flash(f"Design progress email {log.status}.", "info")
    return redirect(url_for("design.board"))


_CSV_COLUMNS = [
    "number", "title", "design_name", "windchill_number", "epdm_number",
    "program", "design_authority", "pm", "dm", "dtl", "da_po", "status",
    "req", "ip", "cdr", "dr", "dwg", "tpr", "audit",
    "accepted", "accepted_by", "doc_location", "notes",
]


@bp.route("/export.csv")
@login_required
def export_csv():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(_CSV_COLUMNS)
    for p in DesignProject.query.order_by(DesignProject.number.asc().nullslast()).all():
        writer.writerow([getattr(p, c) if getattr(p, c) is not None else "" for c in _CSV_COLUMNS])
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=design_projects.csv"},
    )


@bp.route("/import", methods=["GET", "POST"])
@login_required
@editor_required
def import_csv():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename:
            flash("Please choose a CSV file.", "danger")
            return redirect(url_for("design.import_csv"))
        try:
            text = file.read().decode("utf-8-sig")
        except UnicodeDecodeError:
            text = file.read().decode("latin-1")
        reader = csv.DictReader(io.StringIO(text))
        added = 0
        for row in reader:
            if not (row.get("title") or "").strip():
                continue
            dp = DesignProject(title=row["title"].strip())
            dp.number = int(row["number"]) if (row.get("number") or "").strip().isdigit() else None
            for field in ["design_name", "windchill_number", "epdm_number", "program",
                          "design_authority", "pm", "dm", "dtl", "da_po", "status",
                          "accepted", "accepted_by", "doc_location", "notes"]:
                setattr(dp, field, (row.get(field) or "").strip() or None)
            if not dp.status:
                dp.status = "A"
            for field in _INT_FIELDS:
                setattr(dp, field, _coerce_int(row.get(field)))
            db.session.add(dp)
            added += 1
        db.session.commit()
        flash(f"Imported {added} design project(s).", "success")
        return redirect(url_for("design.board"))
    return render_template("design/import.html", columns=_CSV_COLUMNS)
