"""Preventive maintenance scheduling."""

from __future__ import annotations

from datetime import datetime, timedelta

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import login_required

from ..extensions import db
from ..models import WO_PRIORITIES, Asset, PMSchedule, User, WorkOrder
from ..permissions import editor_required

bp = Blueprint("pm", __name__, url_prefix="/pm")


def _parse_date(value):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


@bp.route("/")
@login_required
def list_pm():
    schedules = PMSchedule.query.order_by(PMSchedule.next_due.asc().nullslast()).all()
    due_now = [p for p in schedules if p.is_due]
    return render_template("pm/list.html", schedules=schedules, due_count=len(due_now))


@bp.route("/new", methods=["GET", "POST"])
@bp.route("/<int:pm_id>/edit", methods=["GET", "POST"])
@login_required
@editor_required
def edit(pm_id=None):
    pm = db.get_or_404(PMSchedule, pm_id) if pm_id else None
    if request.method == "POST":
        f = request.form
        if pm is None:
            pm = PMSchedule()
            db.session.add(pm)
        pm.title = f.get("title", "").strip()
        pm.description = f.get("description", "")
        pm.asset_id = int(f["asset_id"]) if f.get("asset_id") else None
        pm.assignee_id = int(f["assignee_id"]) if f.get("assignee_id") else None
        pm.frequency_days = int(f.get("frequency_days") or 30)
        pm.priority = f.get("priority", "medium")
        pm.estimated_hours = float(f.get("estimated_hours") or 1)
        pm.next_due = _parse_date(f.get("next_due"))
        pm.active = f.get("active") == "on"
        if not pm.title:
            flash("Title is required.", "danger")
        else:
            db.session.commit()
            flash("PM schedule saved.", "success")
            return redirect(url_for("pm.list_pm"))
    return render_template(
        "pm/edit.html", pm=pm,
        assets=Asset.query.order_by(Asset.tag).all(),
        users=User.query.order_by(User.full_name).all(),
        priorities=WO_PRIORITIES,
    )


@bp.route("/generate", methods=["POST"])
@login_required
@editor_required
def generate():
    """Create work orders for every PM whose next-due date has arrived."""
    due = [p for p in PMSchedule.query.filter_by(active=True).all() if p.is_due]
    created = 0
    for pm in due:
        last = WorkOrder.query.order_by(WorkOrder.id.desc()).first()
        number = f"WO-{1000 + (last.id if last else 0) + 1}"
        wo = WorkOrder(
            number=number,
            title=f"[PM] {pm.title}",
            description=pm.description,
            wo_type="preventive",
            priority=pm.priority,
            status="open",
            asset_id=pm.asset_id,
            assignee_id=pm.assignee_id,
            pm_schedule_id=pm.id,
            requested_by="PM Scheduler",
            due_date=datetime.utcnow() + timedelta(days=7),
            labor_hours=0.0,
        )
        db.session.add(wo)
        pm.last_generated = datetime.utcnow().date()
        pm.next_due = datetime.utcnow().date() + timedelta(days=pm.frequency_days)
        created += 1
    db.session.commit()
    flash(f"Generated {created} preventive work order(s).", "success" if created else "info")
    return redirect(url_for("pm.list_pm"))
