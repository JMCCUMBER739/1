"""Work order management — the heart of the CMMS."""

from __future__ import annotations

from datetime import datetime

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import login_required

from .. import email_service
from ..extensions import db
from ..models import (
    OPEN_WO_STATUSES,
    WO_PRIORITIES,
    WO_STATUSES,
    WO_TYPES,
    Asset,
    Document,
    User,
    WorkOrder,
)
from ..permissions import editor_required

bp = Blueprint("workorders", __name__, url_prefix="/workorders")


def _parse_dt(value):
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _next_number() -> str:
    last = WorkOrder.query.order_by(WorkOrder.id.desc()).first()
    base = 1000 + (last.id if last else 0) + 1
    return f"WO-{base}"


@bp.route("/")
@login_required
def list_workorders():
    status = request.args.get("status", "")
    priority = request.args.get("priority", "")
    wo_type = request.args.get("type", "")
    assignee = request.args.get("assignee", "")
    view_mode = request.args.get("view", "")

    query = WorkOrder.query
    if view_mode == "open":
        query = query.filter(WorkOrder.status.in_(OPEN_WO_STATUSES))
    if status:
        query = query.filter_by(status=status)
    if priority:
        query = query.filter_by(priority=priority)
    if wo_type:
        query = query.filter_by(wo_type=wo_type)
    if assignee:
        query = query.filter_by(assignee_id=int(assignee))

    workorders = query.order_by(WorkOrder.created_at.desc()).all()
    if view_mode == "overdue":
        workorders = [w for w in workorders if w.is_overdue]

    return render_template(
        "workorders/list.html", workorders=workorders,
        statuses=WO_STATUSES, priorities=WO_PRIORITIES, types=WO_TYPES,
        users=User.query.order_by(User.full_name).all(),
        status=status, priority=priority, wo_type=wo_type,
        assignee=assignee, view_mode=view_mode,
    )


@bp.route("/<int:wo_id>")
@login_required
def view(wo_id):
    wo = db.get_or_404(WorkOrder, wo_id)
    docs = Document.query.filter_by(workorder_id=wo.id).all()
    return render_template("workorders/view.html", wo=wo, docs=docs,
                           statuses=WO_STATUSES)


@bp.route("/new", methods=["GET", "POST"])
@bp.route("/<int:wo_id>/edit", methods=["GET", "POST"])
@login_required
@editor_required
def edit(wo_id=None):
    wo = db.get_or_404(WorkOrder, wo_id) if wo_id else None

    if request.method == "POST":
        f = request.form
        is_new = wo is None
        if is_new:
            wo = WorkOrder(number=_next_number())
            db.session.add(wo)
        previous_assignee = None if is_new else wo.assignee_id

        wo.title = f.get("title", "").strip()
        wo.description = f.get("description", "")
        wo.wo_type = f.get("wo_type", "corrective")
        wo.priority = f.get("priority", "medium")
        wo.status = f.get("status", "open")
        wo.asset_id = int(f["asset_id"]) if f.get("asset_id") else None
        wo.assignee_id = int(f["assignee_id"]) if f.get("assignee_id") else None
        wo.requested_by = f.get("requested_by", "")
        wo.due_date = _parse_dt(f.get("due_date"))
        wo.labor_hours = float(f.get("labor_hours") or 0)
        wo.downtime_hours = float(f.get("downtime_hours") or 0)
        wo.cost_parts = float(f.get("cost_parts") or 0)
        wo.cost_labor = float(f.get("cost_labor") or 0)
        wo.failure_code = f.get("failure_code", "")
        wo.resolution = f.get("resolution", "")

        if wo.status == "in_progress" and not wo.started_at:
            wo.started_at = datetime.utcnow()
        if wo.status == "completed" and not wo.completed_at:
            wo.completed_at = datetime.utcnow()

        if not wo.title:
            flash("Title is required.", "danger")
        else:
            db.session.commit()
            # Auto-notify on assignment changes or new high-priority orders.
            if wo.assignee_id and (is_new or wo.assignee_id != previous_assignee):
                if f.get("notify") == "on":
                    log = email_service.notify_workorder_event(wo)
                    if log:
                        flash(f"Notification {log.status} to assignee.", "info")
            flash("Work order saved.", "success")
            return redirect(url_for("workorders.view", wo_id=wo.id))

    return render_template(
        "workorders/edit.html", wo=wo,
        assets=Asset.query.order_by(Asset.tag).all(),
        users=User.query.order_by(User.full_name).all(),
        statuses=WO_STATUSES, priorities=WO_PRIORITIES, types=WO_TYPES,
    )


@bp.route("/<int:wo_id>/status", methods=["POST"])
@login_required
@editor_required
def set_status(wo_id):
    wo = db.get_or_404(WorkOrder, wo_id)
    new_status = request.form.get("status")
    if new_status in WO_STATUSES:
        wo.status = new_status
        if new_status == "in_progress" and not wo.started_at:
            wo.started_at = datetime.utcnow()
        if new_status == "completed" and not wo.completed_at:
            wo.completed_at = datetime.utcnow()
        db.session.commit()
        flash(f"Status updated to {new_status}.", "success")
    return redirect(url_for("workorders.view", wo_id=wo.id))


@bp.route("/<int:wo_id>/notify", methods=["POST"])
@login_required
@editor_required
def notify(wo_id):
    wo = db.get_or_404(WorkOrder, wo_id)
    log = email_service.notify_workorder_event(wo)
    if log:
        flash(f"Work order notification {log.status}.", "info")
    else:
        flash("No subscribed recipients found.", "warning")
    return redirect(url_for("workorders.view", wo_id=wo.id))
