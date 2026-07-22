"""Dashboard and advanced analytics views."""

from __future__ import annotations

from datetime import datetime, timedelta

from flask import Blueprint, render_template
from flask_login import login_required

from .. import analytics
from ..models import (
    OPEN_WO_STATUSES,
    Asset,
    MaintenanceRequest,
    PMSchedule,
    Part,
    Project,
    WorkOrder,
)

bp = Blueprint("dashboard", __name__)


@bp.route("/")
@login_required
def index():
    kpis = analytics.dashboard_kpis()

    open_wos = (
        WorkOrder.query.filter(WorkOrder.status.in_(OPEN_WO_STATUSES))
        .order_by(WorkOrder.due_date.asc().nullslast())
        .limit(8)
        .all()
    )
    overdue = [w for w in WorkOrder.query.filter(
        WorkOrder.status.in_(OPEN_WO_STATUSES)).all() if w.is_overdue]

    soon = datetime.utcnow().date() + timedelta(days=21)
    cal_due = sorted(
        [a for a in Asset.query.filter_by(requires_calibration=True).all()
         if a.next_calibration_due and a.next_calibration_due <= soon],
        key=lambda a: a.next_calibration_due,
    )
    pms_due = sorted(
        [p for p in PMSchedule.query.filter_by(active=True).all() if p.is_due],
        key=lambda p: p.next_due,
    )
    low_stock = [p for p in Part.query.all() if p.below_reorder]
    at_risk_projects = Project.query.filter(
        Project.status.in_(["planning", "active", "on_hold"]),
        Project.health.in_(["at_risk", "off_track"]),
    ).all()

    charts = {
        "wo_trend": analytics.workorder_trend(6),
        "wo_breakdowns": analytics.workorder_breakdowns(),
        "asset_status": analytics.asset_status_breakdown(),
        "project": analytics.project_analytics(),
    }

    return render_template(
        "dashboard/index.html",
        kpis=kpis,
        design=analytics.design_metrics(),
        open_wos=open_wos,
        overdue_count=len(overdue),
        cal_due=cal_due,
        pms_due=pms_due,
        low_stock=low_stock,
        at_risk_projects=at_risk_projects,
        new_requests=MaintenanceRequest.query.filter_by(status="new").count(),
        charts=charts,
    )


@bp.route("/analytics")
@login_required
def advanced():
    charts = {
        "wo_trend": analytics.workorder_trend(12),
        "cost_trend": analytics.cost_trend(12),
        "wo_breakdowns": analytics.workorder_breakdowns(),
        "asset_status": analytics.asset_status_breakdown(),
        "downtime": analytics.downtime_by_asset(),
        "workload": analytics.workload_by_assignee(),
        "project": analytics.project_analytics(),
    }
    return render_template(
        "dashboard/analytics.html",
        kpis=analytics.dashboard_kpis(),
        charts=charts,
    )
