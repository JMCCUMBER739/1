"""Maintenance requests (manual entry or simulated email intake)."""

from __future__ import annotations

from datetime import datetime, timedelta

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from .. import email_service
from ..extensions import db
from ..models import (
    WO_PRIORITIES,
    Asset,
    MaintenanceRequest,
    WorkOrder,
)
from ..permissions import editor_required

bp = Blueprint("requests", __name__, url_prefix="/requests")


@bp.route("/")
@login_required
def list_requests():
    status = request.args.get("status", "")
    query = MaintenanceRequest.query
    if status:
        query = query.filter_by(status=status)
    items = query.order_by(MaintenanceRequest.created_at.desc()).all()
    return render_template("requests/list.html", items=items, status=status,
                           assets=Asset.query.order_by(Asset.tag).all(),
                           priorities=WO_PRIORITIES)


@bp.route("/new", methods=["POST"])
@login_required
def create():
    f = request.form
    source = f.get("source", "manual")
    req = MaintenanceRequest(
        subject=f.get("subject", "").strip(),
        body=f.get("body", ""),
        requester_name=f.get("requester_name", ""),
        requester_email=f.get("requester_email", ""),
        source="email" if source == "email" else "manual",
        priority=f.get("priority", "medium"),
        asset_id=int(f["asset_id"]) if f.get("asset_id") else None,
        status="new",
    )
    if not req.subject:
        flash("Subject is required.", "danger")
        return redirect(url_for("requests.list_requests"))
    db.session.add(req)
    db.session.commit()
    if f.get("notify") == "on":
        email_service.notify_request_received(req)
    flash("Request logged.", "success")
    return redirect(url_for("requests.list_requests"))


@bp.route("/<int:req_id>/convert", methods=["POST"])
@login_required
@editor_required
def convert(req_id):
    req = db.get_or_404(MaintenanceRequest, req_id)
    if req.workorder_id:
        flash("Request already converted.", "info")
        return redirect(url_for("requests.list_requests"))

    last = WorkOrder.query.order_by(WorkOrder.id.desc()).first()
    number = f"WO-{1000 + (last.id if last else 0) + 1}"
    wo = WorkOrder(
        number=number,
        title=req.subject,
        description=req.body,
        wo_type="corrective",
        priority=req.priority,
        status="open",
        asset_id=req.asset_id,
        requested_by=req.requester_email or req.requester_name,
        due_date=datetime.utcnow() + timedelta(days=7),
    )
    db.session.add(wo)
    db.session.flush()
    req.status = "converted"
    req.workorder_id = wo.id
    db.session.commit()
    flash(f"Converted to work order {wo.number}.", "success")
    return redirect(url_for("workorders.view", wo_id=wo.id))


@bp.route("/<int:req_id>/reject", methods=["POST"])
@login_required
@editor_required
def reject(req_id):
    req = db.get_or_404(MaintenanceRequest, req_id)
    req.status = "rejected"
    db.session.commit()
    flash("Request rejected.", "info")
    return redirect(url_for("requests.list_requests"))
