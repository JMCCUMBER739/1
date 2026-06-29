"""Engineering projects & data tracking."""

from __future__ import annotations

from datetime import datetime

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from .. import email_service
from ..extensions import db
from ..models import (
    DISCIPLINES,
    PROJECT_HEALTH,
    PROJECT_STATUSES,
    PROJECT_TYPES,
    Document,
    EmailTemplate,
    Project,
    ProjectUpdate,
    User,
)
from ..permissions import editor_required

bp = Blueprint("projects", __name__, url_prefix="/projects")


def _parse_date(value):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


@bp.route("/")
@login_required
def list_projects():
    status = request.args.get("status", "")
    discipline = request.args.get("discipline", "")
    query = Project.query
    if status:
        query = query.filter_by(status=status)
    if discipline:
        query = query.filter_by(discipline=discipline)
    projects = query.order_by(Project.code).all()
    return render_template(
        "projects/list.html", projects=projects,
        statuses=PROJECT_STATUSES, disciplines=DISCIPLINES,
        status=status, discipline=discipline,
    )


@bp.route("/<int:project_id>")
@login_required
def view(project_id):
    project = db.get_or_404(Project, project_id)
    docs = Document.query.filter_by(project_id=project.id).all()
    # Build a simple metric time series from logged metric updates.
    metric_series: dict[str, list] = {}
    for upd in sorted(project.updates, key=lambda u: u.created_at):
        if upd.update_type == "metric" and upd.metric_name and upd.metric_value is not None:
            metric_series.setdefault(upd.metric_name, {"labels": [], "values": []})
            metric_series[upd.metric_name]["labels"].append(upd.created_at.strftime("%m/%d"))
            metric_series[upd.metric_name]["values"].append(upd.metric_value)
    return render_template("projects/view.html", project=project, docs=docs,
                           metric_series=metric_series)


@bp.route("/new", methods=["GET", "POST"])
@bp.route("/<int:project_id>/edit", methods=["GET", "POST"])
@login_required
@editor_required
def edit(project_id=None):
    project = db.get_or_404(Project, project_id) if project_id else None
    if request.method == "POST":
        f = request.form
        if project is None:
            project = Project()
            db.session.add(project)
        project.code = f.get("code", "").strip()
        project.name = f.get("name", "").strip()
        project.description = f.get("description", "")
        project.project_type = f.get("project_type", "rnd")
        project.discipline = f.get("discipline", "other")
        project.status = f.get("status", "active")
        project.health = f.get("health", "on_track")
        project.lead_id = int(f["lead_id"]) if f.get("lead_id") else None
        project.percent_complete = int(f.get("percent_complete") or 0)
        project.budget = float(f.get("budget") or 0)
        project.spent = float(f.get("spent") or 0)
        project.start_date = _parse_date(f.get("start_date"))
        project.target_date = _parse_date(f.get("target_date"))
        if not project.name:
            flash("Project name is required.", "danger")
        else:
            db.session.commit()
            flash("Project saved.", "success")
            return redirect(url_for("projects.view", project_id=project.id))
    return render_template(
        "projects/edit.html", project=project,
        users=User.query.order_by(User.full_name).all(),
        statuses=PROJECT_STATUSES, healths=PROJECT_HEALTH,
        types=PROJECT_TYPES, disciplines=DISCIPLINES,
    )


@bp.route("/<int:project_id>/update", methods=["POST"])
@login_required
@editor_required
def add_update(project_id):
    project = db.get_or_404(Project, project_id)
    f = request.form
    upd = ProjectUpdate(
        project_id=project.id,
        author_id=current_user.id,
        update_type=f.get("update_type", "note"),
        title=f.get("title", ""),
        body=f.get("body", ""),
        metric_name=f.get("metric_name", "") or None,
        metric_value=float(f["metric_value"]) if f.get("metric_value") else None,
        metric_unit=f.get("metric_unit", ""),
        percent_complete=int(f["percent_complete"]) if f.get("percent_complete") else None,
    )
    db.session.add(upd)
    if upd.percent_complete is not None:
        project.percent_complete = upd.percent_complete
    db.session.commit()
    flash("Project update logged.", "success")
    return redirect(url_for("projects.view", project_id=project.id))


@bp.route("/<int:project_id>/email", methods=["POST"])
@login_required
@editor_required
def email_status(project_id):
    project = db.get_or_404(Project, project_id)
    template = EmailTemplate.query.filter_by(key="project_status").first()
    ctx = {
        **email_service.base_context(),
        "project_code": project.code,
        "project_name": project.name,
        "project_type": project.project_type,
        "lead": project.lead.full_name if project.lead else "TBD",
        "health": project.health,
        "percent_complete": project.percent_complete,
        "target_date": project.target_date.strftime("%Y-%m-%d") if project.target_date else "TBD",
        "summary": request.form.get("summary", project.description or ""),
    }
    recipients = email_service.recipients_for("project")
    if template:
        subject, body = email_service.render_template_record(template, ctx)
    else:
        subject = f"Project {project.code} status"
        body = ctx["summary"]
    log = email_service.send_email(subject, recipients, body, category="project")
    flash(f"Project status email {log.status}.", "info")
    return redirect(url_for("projects.view", project_id=project.id))
