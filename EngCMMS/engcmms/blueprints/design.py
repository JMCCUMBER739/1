"""Engineering Design module — CD-8000.002 focused.

Combines:
  * Configuration-management progress board (maturity matrix)
  * Phase-gate project hub (Requirements → Conceptual → Detailed → Acceptance)
  * Requirements Document + Requirements Verification Matrix
  * Design reviews + action items
  * Calculations tracking
  * Paperwork reminders / email notifications
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta

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
    DESIGN_PHASES,
    DESIGN_PROJECT_STATUSES,
    DESIGN_RIGORS,
    DESIGN_STAGE_CODES,
    DESIGN_STAGES,
    DESIGN_TYPES,
    DOC_LOCATIONS,
    FUNCTIONAL_CLASSIFICATIONS,
    PAPERWORK_TYPES,
    REQUIREMENT_CATEGORIES,
    REQUIREMENT_CATEGORY_PREFIX,
    REQUIREMENT_STATUSES,
    REVIEW_FORMALITIES,
    REVIEW_OUTCOMES,
    REVIEW_TYPES,
    VERIFICATION_METHODS,
    DesignActionItem,
    DesignCalculation,
    DesignProject,
    DesignRequirement,
    DesignReview,
    PaperworkReminder,
    VerificationItem,
)
from ..permissions import editor_required

bp = Blueprint("design", __name__, url_prefix="/design")

_INT_FIELDS = DesignProject.STAGE_FIELDS
_TEXT_FIELDS = [
    "title", "design_name", "windchill_number", "epdm_number", "program",
    "doc_location", "design_authority", "pm", "dm", "dtl", "da_po", "status",
    "accepted", "accepted_by", "notes", "phase", "rigor",
    "functional_classification", "customer", "stakeholders", "scope_statement",
    "cm_system", "design_software", "acceptance_criteria",
]


def _coerce_int(value, default=0):
    try:
        return max(0, min(5, int(value)))
    except (TypeError, ValueError):
        return default


def _parse_date(value):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _parse_dt(value):
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Design Engineering Dashboard (CD-8000.002 KPIs)
# ---------------------------------------------------------------------------
@bp.route("/dashboard")
@login_required
def eng_dashboard():
    from .. import analytics

    metrics = analytics.design_metrics()
    overdue_reminders = [
        r for r in PaperworkReminder.query.filter_by(status="open").all() if r.is_overdue
    ]
    open_actions = [
        a for a in DesignActionItem.query.filter_by(status="open").all()
    ]
    overdue_actions = [a for a in open_actions if a.is_overdue]
    upcoming = (
        PaperworkReminder.query.filter(
            PaperworkReminder.status == "open",
            PaperworkReminder.due_date.isnot(None),
            PaperworkReminder.due_date <= (datetime.utcnow().date() + timedelta(days=14)),
        ).order_by(PaperworkReminder.due_date.asc()).limit(10).all()
    )
    recent_reviews = (
        DesignReview.query.order_by(DesignReview.held_at.desc()).limit(8).all()
    )
    at_risk = DesignProject.query.filter(
        DesignProject.status == "A",
        DesignProject.functional_classification.in_(["SC", "SS"]),
    ).all()
    phase_stuck = [
        p for p in DesignProject.query.filter_by(status="A").all()
        if p.phase_entered_at and (datetime.utcnow() - p.phase_entered_at).days > 60
    ]
    return render_template(
        "design/eng_dashboard.html",
        metrics=metrics,
        overdue_reminders=overdue_reminders,
        overdue_actions=overdue_actions,
        upcoming=upcoming,
        recent_reviews=recent_reviews,
        at_risk=at_risk,
        phase_stuck=phase_stuck,
        phases=DESIGN_PHASES,
    )


# ---------------------------------------------------------------------------
# Progress board (existing matrix)
# ---------------------------------------------------------------------------
@bp.route("/")
@login_required
def board():
    status = request.args.get("status", "")
    program = request.args.get("program", "")
    phase = request.args.get("phase", "")
    q = request.args.get("q", "").strip()

    query = DesignProject.query
    if status:
        query = query.filter_by(status=status)
    if program:
        query = query.filter_by(program=program)
    if phase:
        query = query.filter_by(phase=phase)
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
        phases=DESIGN_PHASES,
        status=status, program=program, q=q, phase=phase,
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
            dp = DesignProject(phase_entered_at=datetime.utcnow())
            db.session.add(dp)
        old_phase = dp.phase
        dp.number = int(f["number"]) if f.get("number") else None
        dp.title = f.get("title", "").strip()
        for field in _TEXT_FIELDS:
            if field == "title":
                continue
            setattr(dp, field, (f.get(field, "") or "").strip() or None)
        if not dp.status:
            dp.status = "A"
        if not dp.phase:
            dp.phase = "requirements"
        if not dp.rigor:
            dp.rigor = "medium"
        if not dp.functional_classification:
            dp.functional_classification = "GS"
        if dp.phase != old_phase:
            dp.phase_entered_at = datetime.utcnow()
        dp.target_completion = _parse_date(f.get("target_completion"))
        for field in _INT_FIELDS:
            setattr(dp, field, _coerce_int(f.get(field)))
        if not dp.title:
            flash("Title is required.", "danger")
        else:
            db.session.commit()
            flash("Design project saved.", "success")
            return redirect(url_for("design.hub", dp_id=dp.id))

    return render_template(
        "design/edit.html", dp=dp, stages=DESIGN_STAGES,
        stage_codes=DESIGN_STAGE_CODES, statuses=DESIGN_PROJECT_STATUSES,
        doc_locations=DOC_LOCATIONS, types=DESIGN_TYPES,
        phases=DESIGN_PHASES, rigors=DESIGN_RIGORS,
        classifications=FUNCTIONAL_CLASSIFICATIONS,
    )


@bp.route("/<int:dp_id>")
@login_required
def hub(dp_id):
    """Project hub — CD-8000.002 phase view for one design."""
    dp = db.get_or_404(DesignProject, dp_id)
    return render_template(
        "design/hub.html", dp=dp, stages=DESIGN_STAGES,
        stage_codes=DESIGN_STAGE_CODES, phases=DESIGN_PHASES,
        review_types=REVIEW_TYPES, paperwork_types=PAPERWORK_TYPES,
    )


@bp.route("/<int:dp_id>/cell", methods=["POST"])
@login_required
@editor_required
def update_cell(dp_id):
    dp = db.get_or_404(DesignProject, dp_id)
    field = request.form.get("field", "")
    value = request.form.get("value", "")

    if field in _INT_FIELDS:
        setattr(dp, field, _coerce_int(value))
    elif field in ("status", "accepted", "da_po", "doc_location", "phase", "rigor",
                   "functional_classification"):
        old = getattr(dp, field)
        setattr(dp, field, value)
        if field == "phase" and value != old:
            dp.phase_entered_at = datetime.utcnow()
    else:
        abort(400)
    db.session.commit()

    new_value = getattr(dp, field)
    return jsonify({
        "ok": True, "field": field, "value": new_value,
        "maturity": dp.maturity_pct,
        "code_class": f"code-{new_value}" if field in _INT_FIELDS else "",
    })


@bp.route("/<int:dp_id>/phase", methods=["POST"])
@login_required
@editor_required
def advance_phase(dp_id):
    dp = db.get_or_404(DesignProject, dp_id)
    phase_keys = [p[0] for p in DESIGN_PHASES]
    try:
        idx = phase_keys.index(dp.phase or "requirements")
    except ValueError:
        idx = 0
    direction = request.form.get("direction", "next")
    if direction == "next" and idx < len(phase_keys) - 1:
        dp.phase = phase_keys[idx + 1]
        dp.phase_entered_at = datetime.utcnow()
        flash(f"Advanced to {dp.phase_label}.", "success")
    elif direction == "prev" and idx > 0:
        dp.phase = phase_keys[idx - 1]
        dp.phase_entered_at = datetime.utcnow()
        flash(f"Moved back to {dp.phase_label}.", "info")
    db.session.commit()
    return redirect(url_for("design.hub", dp_id=dp.id))


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
    subject = f"Design progress: {dp.title} — {dp.phase_label} ({dp.maturity_pct:.0f}%)"
    lines = [
        f"Design project: {dp.title}",
        f"Design name: {dp.design_name or '—'}",
        f"Phase: {dp.phase_label}   Rigor: {dp.rigor_label}   Class: {dp.classification_label}",
        f"WindChill #: {dp.windchill_number or '—'}   ePDM #: {dp.epdm_number or '—'}",
        f"PM: {dp.pm or '—'}   DM: {dp.dm or '—'}   DTL: {dp.dtl or '—'}",
        "",
    ]
    for field, abbr, name in DESIGN_STAGES:
        code = getattr(dp, field)
        lines.append(f"  {abbr:6} {name:26} {code} ({DESIGN_STAGE_CODES.get(code, '')})")
    lines += ["", f"Overall maturity: {dp.maturity_pct:.0f}%",
              f"Open review actions: {dp.open_action_count}",
              f"Acceptance: {dp.accept_display}"]
    log = email_service.send_email(subject, recipients, "\n".join(lines), category="project")
    dp.email_sent = True
    dp.email_sent_at = datetime.utcnow()
    db.session.commit()
    flash(f"Design progress email {log.status}.", "info")
    return redirect(request.referrer or url_for("design.hub", dp_id=dp.id))


# ---------------------------------------------------------------------------
# Requirements Document + Verification Matrix
# ---------------------------------------------------------------------------
@bp.route("/<int:dp_id>/requirements", methods=["GET", "POST"])
@login_required
def requirements(dp_id):
    dp = db.get_or_404(DesignProject, dp_id)
    if request.method == "POST":
        if not current_user.can_edit:
            abort(403)
        f = request.form
        category = f.get("category", "technical")
        prefix = REQUIREMENT_CATEGORY_PREFIX.get(category, "O")
        existing = DesignRequirement.query.filter_by(
            design_project_id=dp.id, category=category
        ).count()
        req_number = f.get("req_number", "").strip() or f"{prefix}-{existing + 1}"
        req = DesignRequirement(
            design_project_id=dp.id,
            req_number=req_number,
            category=category,
            source=f.get("source", ""),
            statement=f.get("statement", "").strip(),
            status=f.get("status", "draft"),
        )
        if not req.statement:
            flash("Requirement statement is required.", "danger")
        else:
            db.session.add(req)
            db.session.flush()
            db.session.add(VerificationItem(
                requirement_id=req.id,
                method=f.get("method", "test"),
                success_criteria=f.get("success_criteria", ""),
            ))
            if (dp.req or 0) < 1:
                dp.req = 1
            db.session.commit()
            flash(f"Requirement {req.req_number} added.", "success")
        return redirect(url_for("design.requirements", dp_id=dp.id))

    return render_template(
        "design/requirements.html", dp=dp,
        categories=REQUIREMENT_CATEGORIES, statuses=REQUIREMENT_STATUSES,
        methods=VERIFICATION_METHODS,
    )


@bp.route("/requirements/<int:req_id>/update", methods=["POST"])
@login_required
@editor_required
def update_requirement(req_id):
    req = db.get_or_404(DesignRequirement, req_id)
    f = request.form
    req.statement = f.get("statement", req.statement)
    req.source = f.get("source", req.source)
    req.status = f.get("status", req.status)
    req.category = f.get("category", req.category)
    if req.verification:
        req.verification.success_criteria = f.get("success_criteria", req.verification.success_criteria)
        req.verification.method = f.get("method", req.verification.method)
        req.verification.performer = f.get("performer", req.verification.performer)
        req.verification.result = f.get("result", req.verification.result)
        req.verification.notes = f.get("notes", req.verification.notes)
        req.verification.verified_at = _parse_date(f.get("verified_at"))
    db.session.commit()
    flash("Requirement updated.", "success")
    return redirect(url_for("design.requirements", dp_id=req.design_project_id))


@bp.route("/requirements/<int:req_id>/delete", methods=["POST"])
@login_required
@editor_required
def delete_requirement(req_id):
    req = db.get_or_404(DesignRequirement, req_id)
    dp_id = req.design_project_id
    db.session.delete(req)
    db.session.commit()
    flash("Requirement deleted.", "info")
    return redirect(url_for("design.requirements", dp_id=dp_id))


@bp.route("/<int:dp_id>/rvm")
@login_required
def rvm(dp_id):
    dp = db.get_or_404(DesignProject, dp_id)
    return render_template("design/rvm.html", dp=dp, methods=VERIFICATION_METHODS)


# ---------------------------------------------------------------------------
# Design Reviews + Action Items
# ---------------------------------------------------------------------------
@bp.route("/<int:dp_id>/reviews", methods=["GET", "POST"])
@login_required
def reviews(dp_id):
    dp = db.get_or_404(DesignProject, dp_id)
    if request.method == "POST":
        if not current_user.can_edit:
            abort(403)
        f = request.form
        review = DesignReview(
            design_project_id=dp.id,
            review_type=f.get("review_type", "interim"),
            formality=f.get("formality", "document"),
            title=f.get("title", "").strip() or f.get("review_type", "Review"),
            held_at=_parse_dt(f.get("held_at")) or datetime.utcnow(),
            location=f.get("location", ""),
            attendees=f.get("attendees", ""),
            independent_reviewer=f.get("independent_reviewer", ""),
            documents_presented=f.get("documents_presented", ""),
            topics=f.get("topics", ""),
            decisions=f.get("decisions", ""),
            outcome=f.get("outcome", "pending"),
            requirements_revised=f.get("requirements_revised") == "on",
            notes=f.get("notes", ""),
        )
        db.session.add(review)
        db.session.flush()
        # Optional first action item
        if f.get("action_description", "").strip():
            db.session.add(DesignActionItem(
                review_id=review.id,
                description=f.get("action_description").strip(),
                assignee=f.get("action_assignee", ""),
                due_date=_parse_date(f.get("action_due")),
            ))
        # Bump CDR/DR maturity when a review is logged
        if review.review_type == "conceptual" and (dp.cdr or 0) < 2:
            dp.cdr = max(dp.cdr or 0, 2)
        elif review.review_type in ("interim", "final") and (dp.dr or 0) < 2:
            dp.dr = max(dp.dr or 0, 2)
        db.session.commit()
        flash("Design review recorded.", "success")
        return redirect(url_for("design.reviews", dp_id=dp.id))

    return render_template(
        "design/reviews.html", dp=dp,
        review_types=REVIEW_TYPES, formalities=REVIEW_FORMALITIES,
        outcomes=REVIEW_OUTCOMES,
    )


@bp.route("/reviews/<int:review_id>/action", methods=["POST"])
@login_required
@editor_required
def add_action(review_id):
    review = db.get_or_404(DesignReview, review_id)
    f = request.form
    desc = f.get("description", "").strip()
    if not desc:
        flash("Action description required.", "danger")
    else:
        db.session.add(DesignActionItem(
            review_id=review.id, description=desc,
            assignee=f.get("assignee", ""),
            due_date=_parse_date(f.get("due_date")),
        ))
        db.session.commit()
        flash("Action item added.", "success")
    return redirect(url_for("design.reviews", dp_id=review.design_project_id))


@bp.route("/actions/<int:action_id>/close", methods=["POST"])
@login_required
@editor_required
def close_action(action_id):
    action = db.get_or_404(DesignActionItem, action_id)
    action.status = "closed"
    action.closed_at = datetime.utcnow()
    db.session.commit()
    flash("Action item closed.", "success")
    return redirect(url_for("design.reviews", dp_id=action.review.design_project_id))


# ---------------------------------------------------------------------------
# Calculations
# ---------------------------------------------------------------------------
@bp.route("/<int:dp_id>/calculations", methods=["GET", "POST"])
@login_required
def calculations(dp_id):
    dp = db.get_or_404(DesignProject, dp_id)
    if request.method == "POST":
        if not current_user.can_edit:
            abort(403)
        f = request.form
        calc = DesignCalculation(
            design_project_id=dp.id,
            title=f.get("title", "").strip(),
            preparer=f.get("preparer", ""),
            checker=f.get("checker", ""),
            purpose=f.get("purpose", ""),
            software_used=f.get("software_used", ""),
            status=f.get("status", "draft"),
        )
        if not calc.title:
            flash("Title is required.", "danger")
        else:
            if calc.status == "released":
                calc.released_at = datetime.utcnow()
            db.session.add(calc)
            db.session.commit()
            flash("Calculation recorded.", "success")
        return redirect(url_for("design.calculations", dp_id=dp.id))
    return render_template("design/calculations.html", dp=dp)


# ---------------------------------------------------------------------------
# Paperwork Reminders
# ---------------------------------------------------------------------------
@bp.route("/reminders")
@login_required
def reminders():
    status = request.args.get("status", "open")
    query = PaperworkReminder.query
    if status:
        query = query.filter_by(status=status)
    items = query.order_by(PaperworkReminder.due_date.asc().nullslast()).all()
    return render_template(
        "design/reminders.html", items=items, status=status,
        paperwork_types=PAPERWORK_TYPES,
        projects=DesignProject.query.filter_by(status="A").order_by(DesignProject.title).all(),
    )


@bp.route("/reminders/new", methods=["POST"])
@bp.route("/<int:dp_id>/reminders/new", methods=["POST"])
@login_required
@editor_required
def create_reminder(dp_id=None):
    f = request.form
    rem = PaperworkReminder(
        design_project_id=int(f["design_project_id"]) if f.get("design_project_id") else dp_id,
        paperwork_type=f.get("paperwork_type", "OTHER"),
        title=f.get("title", "").strip(),
        description=f.get("description", ""),
        assignee=f.get("assignee", ""),
        assignee_email=f.get("assignee_email", ""),
        due_date=_parse_date(f.get("due_date")),
        created_by_id=current_user.id,
    )
    if not rem.title:
        flash("Reminder title is required.", "danger")
    else:
        db.session.add(rem)
        db.session.commit()
        flash("Reminder created.", "success")
        if f.get("send_now") == "on":
            _send_reminder(rem)
    if rem.design_project_id:
        return redirect(url_for("design.hub", dp_id=rem.design_project_id))
    return redirect(url_for("design.reminders"))


@bp.route("/reminders/<int:rem_id>/send", methods=["POST"])
@login_required
@editor_required
def send_reminder(rem_id):
    rem = db.get_or_404(PaperworkReminder, rem_id)
    _send_reminder(rem)
    flash("Reminder email queued.", "info")
    return redirect(request.referrer or url_for("design.reminders"))


@bp.route("/reminders/<int:rem_id>/complete", methods=["POST"])
@login_required
@editor_required
def complete_reminder(rem_id):
    rem = db.get_or_404(PaperworkReminder, rem_id)
    rem.status = "completed"
    db.session.commit()
    flash("Reminder marked complete.", "success")
    return redirect(request.referrer or url_for("design.reminders"))


@bp.route("/reminders/send-overdue", methods=["POST"])
@login_required
@editor_required
def send_overdue_reminders():
    overdue = [r for r in PaperworkReminder.query.filter_by(status="open").all() if r.is_overdue]
    for rem in overdue:
        _send_reminder(rem)
    flash(f"Sent {len(overdue)} overdue paperwork reminder(s).", "info")
    return redirect(url_for("design.reminders"))


def _send_reminder(rem: PaperworkReminder) -> None:
    recipients = []
    if rem.assignee_email:
        recipients.append(rem.assignee_email)
    recipients.extend(email_service.recipients_for("project"))
    recipients = list(dict.fromkeys(recipients))
    project = rem.design_project.title if rem.design_project else "—"
    due = rem.due_date.strftime("%Y-%m-%d") if rem.due_date else "Not set"
    overdue = " (OVERDUE)" if rem.is_overdue else ""
    subject = f"[Paperwork{overdue}] {rem.type_label}: {rem.title}"
    body = (
        f"This is a reminder that the following design deliverable is needed.\n\n"
        f"Type: {rem.type_label}\n"
        f"Title: {rem.title}\n"
        f"Design project: {project}\n"
        f"Assignee: {rem.assignee or '—'}\n"
        f"Due: {due}{overdue}\n\n"
        f"{rem.description or ''}\n\n"
        f"— {email_service.base_context().get('org_name', 'EngCMMS')} "
        f"(per CD-8000.002 Engineering Design)"
    )
    email_service.send_email(subject, recipients, body, category="project")
    rem.last_sent_at = datetime.utcnow()
    db.session.commit()


# ---------------------------------------------------------------------------
# CSV import / export
# ---------------------------------------------------------------------------
_CSV_COLUMNS = [
    "number", "title", "design_name", "windchill_number", "epdm_number",
    "program", "design_authority", "pm", "dm", "dtl", "da_po", "status",
    "phase", "rigor", "functional_classification", "customer",
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
        output.getvalue(), mimetype="text/csv",
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
            dp = DesignProject(title=row["title"].strip(), phase_entered_at=datetime.utcnow())
            dp.number = int(row["number"]) if (row.get("number") or "").strip().isdigit() else None
            for field in ["design_name", "windchill_number", "epdm_number", "program",
                          "design_authority", "pm", "dm", "dtl", "da_po", "status",
                          "phase", "rigor", "functional_classification", "customer",
                          "accepted", "accepted_by", "doc_location", "notes"]:
                setattr(dp, field, (row.get(field) or "").strip() or None)
            if not dp.status:
                dp.status = "A"
            if not dp.phase:
                dp.phase = "requirements"
            for field in _INT_FIELDS:
                setattr(dp, field, _coerce_int(row.get(field)))
            db.session.add(dp)
            added += 1
        db.session.commit()
        flash(f"Imported {added} design project(s).", "success")
        return redirect(url_for("design.board"))
    return render_template("design/import.html", columns=_CSV_COLUMNS)
