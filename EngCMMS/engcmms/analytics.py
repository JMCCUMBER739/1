"""KPI and advanced-analytics engine.

All functions return plain Python data structures (dicts / lists) so they can be
consumed directly by Jinja templates and serialised to JSON for Chart.js.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timedelta

from sqlalchemy import func

from .extensions import db
from .models import (
    DESIGN_PHASE_LABELS,
    DESIGN_STAGES,
    OPEN_WO_STATUSES,
    Asset,
    DesignActionItem,
    DesignProject,
    DesignRequirement,
    DesignReview,
    PaperworkReminder,
    Part,
    Project,
    VerificationItem,
    WorkOrder,
)


def _month_buckets(months: int = 6):
    """Return ordered (label, year, month) tuples ending with the current month."""
    today = datetime.utcnow().replace(day=1)
    buckets = []
    for i in range(months - 1, -1, -1):
        # Walk backwards i months from the first of this month.
        month = today.month - i
        year = today.year
        while month <= 0:
            month += 12
            year -= 1
        buckets.append((datetime(year, month, 1).strftime("%b %Y"), year, month))
    return buckets


def dashboard_kpis() -> dict:
    """Headline numbers shown as KPI cards at the top of the dashboard."""
    now = datetime.utcnow()
    today = now.date()
    soon = today + timedelta(days=14)

    open_wos = WorkOrder.query.filter(WorkOrder.status.in_(OPEN_WO_STATUSES)).all()
    overdue = [w for w in open_wos if w.is_overdue]

    completed = WorkOrder.query.filter(WorkOrder.status == "completed").all()

    assets = Asset.query.all()
    down_assets = [a for a in assets if a.status == "down"]
    operational = [a for a in assets if a.status == "operational"]

    cal_due = [a for a in assets if a.requires_calibration and a.next_calibration_due
               and a.next_calibration_due <= soon]

    parts = Part.query.all()
    low_stock = [p for p in parts if p.below_reorder]

    projects = Project.query.all()
    active_projects = [p for p in projects if p.status in {"planning", "active", "on_hold"}]
    at_risk = [p for p in active_projects if p.health in {"at_risk", "off_track"}]

    availability = round(len(operational) / len(assets) * 100, 1) if assets else 100.0

    return {
        "open_workorders": len(open_wos),
        "overdue_workorders": len(overdue),
        "completed_workorders": len(completed),
        "total_assets": len(assets),
        "assets_down": len(down_assets),
        "asset_availability": availability,
        "calibrations_due": len(cal_due),
        "low_stock_parts": len(low_stock),
        "inventory_value": round(sum(p.stock_value for p in parts), 2),
        "active_projects": len(active_projects),
        "projects_at_risk": len(at_risk),
        "mttr_hours": mean_time_to_repair(),
        "mtbf_days": mean_time_between_failures(),
        "pm_compliance": pm_compliance_pct(),
    }


def mean_time_to_repair() -> float:
    """Average hours between WO creation and completion for completed jobs."""
    rows = (
        WorkOrder.query.filter(
            WorkOrder.status == "completed",
            WorkOrder.completed_at.isnot(None),
        ).all()
    )
    durations = [w.resolution_hours for w in rows if w.resolution_hours is not None]
    if not durations:
        return 0.0
    return round(sum(durations) / len(durations), 1)


def mean_time_between_failures() -> float:
    """Average days between corrective work orders across all assets."""
    correctives = (
        WorkOrder.query.filter(
            WorkOrder.wo_type == "corrective",
            WorkOrder.asset_id.isnot(None),
        ).order_by(WorkOrder.asset_id, WorkOrder.created_at).all()
    )
    by_asset: dict[int, list[datetime]] = {}
    for wo in correctives:
        by_asset.setdefault(wo.asset_id, []).append(wo.created_at)

    gaps = []
    for dates in by_asset.values():
        for prev, nxt in zip(dates, dates[1:]):
            gaps.append((nxt - prev).total_seconds() / 86400.0)
    if not gaps:
        return 0.0
    return round(sum(gaps) / len(gaps), 1)


def pm_compliance_pct() -> float:
    """Percentage of preventive work orders completed on or before due date."""
    pm_wos = WorkOrder.query.filter(
        WorkOrder.wo_type == "preventive",
        WorkOrder.status == "completed",
    ).all()
    if not pm_wos:
        return 100.0
    on_time = 0
    for wo in pm_wos:
        if wo.due_date is None or (wo.completed_at and wo.completed_at <= wo.due_date):
            on_time += 1
    return round(on_time / len(pm_wos) * 100, 1)


def workorder_trend(months: int = 6) -> dict:
    buckets = _month_buckets(months)
    labels = [b[0] for b in buckets]
    created = []
    completed = []
    for _, year, month in buckets:
        start = datetime(year, month, 1)
        end = datetime(year + (month // 12), (month % 12) + 1, 1)
        created.append(
            WorkOrder.query.filter(
                WorkOrder.created_at >= start, WorkOrder.created_at < end
            ).count()
        )
        completed.append(
            WorkOrder.query.filter(
                WorkOrder.completed_at >= start, WorkOrder.completed_at < end
            ).count()
        )
    return {"labels": labels, "created": created, "completed": completed}


def cost_trend(months: int = 6) -> dict:
    buckets = _month_buckets(months)
    labels = [b[0] for b in buckets]
    parts_cost = []
    labor_cost = []
    for _, year, month in buckets:
        start = datetime(year, month, 1)
        end = datetime(year + (month // 12), (month % 12) + 1, 1)
        rows = WorkOrder.query.filter(
            WorkOrder.completed_at >= start, WorkOrder.completed_at < end
        ).all()
        parts_cost.append(round(sum(w.cost_parts or 0 for w in rows), 2))
        labor_cost.append(round(sum(w.cost_labor or 0 for w in rows), 2))
    return {"labels": labels, "parts": parts_cost, "labor": labor_cost}


def _count_by(column):
    rows = db.session.query(column, func.count()).group_by(column).all()
    return OrderedDict((k or "unknown", v) for k, v in rows)


def workorder_breakdowns() -> dict:
    return {
        "by_status": _count_by(WorkOrder.status),
        "by_priority": _count_by(WorkOrder.priority),
        "by_type": _count_by(WorkOrder.wo_type),
    }


def asset_status_breakdown() -> "OrderedDict":
    return _count_by(Asset.status)


def downtime_by_asset(limit: int = 8) -> dict:
    rows = (
        db.session.query(Asset.name, func.sum(WorkOrder.downtime_hours))
        .join(WorkOrder, WorkOrder.asset_id == Asset.id)
        .group_by(Asset.id)
        .order_by(func.sum(WorkOrder.downtime_hours).desc())
        .limit(limit)
        .all()
    )
    rows = [(name, round(hours or 0, 1)) for name, hours in rows if (hours or 0) > 0]
    return {"labels": [r[0] for r in rows], "values": [r[1] for r in rows]}


def workload_by_assignee() -> dict:
    rows = (
        db.session.query(
            func.coalesce(WorkOrder.assignee_id, 0), func.count()
        )
        .filter(WorkOrder.status.in_(OPEN_WO_STATUSES))
        .group_by(WorkOrder.assignee_id)
        .all()
    )
    from .models import User

    labels, values = [], []
    for assignee_id, count in rows:
        if assignee_id:
            user = db.session.get(User, assignee_id)
            labels.append(user.full_name if user else f"User {assignee_id}")
        else:
            labels.append("Unassigned")
        values.append(count)
    return {"labels": labels, "values": values}


def design_metrics() -> dict:
    """KPIs and chart data for the design-project progress board + CD-8000.002."""
    projects = DesignProject.query.all()
    active = [p for p in projects if p.status == "A"]

    avg_maturity = (
        round(sum(p.maturity_pct for p in active) / len(active), 1) if active else 0.0
    )

    fully_released = 0
    for p in active:
        codes = [getattr(p, f) for f in DesignProject.STAGE_FIELDS]
        relevant = [c for c in codes if c not in (None, 5)]
        if relevant and all(c == 3 for c in relevant):
            fully_released += 1

    pending_accept = len([p for p in active if p.accepted != "Y"])
    emails_sent = len([p for p in projects if p.email_sent])

    stage_labels = [abbr for _, abbr, _ in DESIGN_STAGES]
    stage_values = []
    for field, _abbr, _name in DESIGN_STAGES:
        vals = []
        for p in active:
            code = getattr(p, field)
            if code in (None, 5):
                continue
            vals.append({0: 0, 1: 34, 2: 67, 3: 100, 4: 100}.get(code, 0))
        stage_values.append(round(sum(vals) / len(vals), 0) if vals else 0)

    by_program = OrderedDict()
    for p in active:
        by_program[p.program or "—"] = by_program.get(p.program or "—", 0) + 1

    buckets = OrderedDict([("0-25%", 0), ("26-50%", 0), ("51-75%", 0), ("76-99%", 0), ("100%", 0)])
    for p in active:
        m = p.maturity_pct
        if m >= 100:
            buckets["100%"] += 1
        elif m >= 76:
            buckets["76-99%"] += 1
        elif m >= 51:
            buckets["51-75%"] += 1
        elif m >= 26:
            buckets["26-50%"] += 1
        else:
            buckets["0-25%"] += 1

    # CD-8000.002 phase / rigor / classification breakdowns
    by_phase = OrderedDict((label, 0) for label in DESIGN_PHASE_LABELS.values())
    by_rigor = OrderedDict([("Casual", 0), ("Low", 0), ("Medium", 0), ("High", 0)])
    by_class = OrderedDict([("GS", 0), ("SS", 0), ("SC", 0)])
    for p in active:
        by_phase[p.phase_label] = by_phase.get(p.phase_label, 0) + 1
        by_rigor[p.rigor_label] = by_rigor.get(p.rigor_label, 0) + 1
        fc = p.functional_classification or "GS"
        by_class[fc] = by_class.get(fc, 0) + 1

    sc_ss = by_class.get("SC", 0) + by_class.get("SS", 0)
    high_rigor = by_rigor.get("High", 0)

    open_actions = DesignActionItem.query.filter_by(status="open").count()
    overdue_actions = sum(
        1 for a in DesignActionItem.query.filter_by(status="open").all() if a.is_overdue
    )
    open_reminders = PaperworkReminder.query.filter_by(status="open").count()
    overdue_reminders = sum(
        1 for r in PaperworkReminder.query.filter_by(status="open").all() if r.is_overdue
    )

    total_reqs = DesignRequirement.query.filter(
        DesignRequirement.status != "superseded"
    ).count()
    verified = (
        db.session.query(VerificationItem)
        .filter(VerificationItem.result == "pass")
        .count()
    )
    req_verified_pct = round(verified / total_reqs * 100, 1) if total_reqs else 0.0

    reviews_held = DesignReview.query.count()
    reviews_pending = DesignReview.query.filter_by(outcome="pending").count()

    # Paperwork completeness heuristic based on stage codes
    paperwork_due = []
    for p in active:
        if (p.ip or 0) < 2:
            paperwork_due.append(("IP", p))
        if (p.req or 0) < 2:
            paperwork_due.append(("REQ", p))
        if p.phase in ("conceptual", "detailed", "acceptance") and (p.cdr or 0) < 2 and (p.cdr or 0) != 5:
            paperwork_due.append(("CDR", p))
        if p.phase in ("detailed", "acceptance") and (p.dr or 0) < 2 and (p.dr or 0) != 5:
            paperwork_due.append(("DR", p))

    return {
        "total": len(projects),
        "active": len(active),
        "inactive": len(projects) - len(active),
        "avg_maturity": avg_maturity,
        "fully_released": fully_released,
        "pending_accept": pending_accept,
        "emails_sent": emails_sent,
        "stage_chart": {"labels": stage_labels, "values": stage_values},
        "by_program": by_program,
        "maturity_dist": buckets,
        "by_phase": by_phase,
        "by_rigor": by_rigor,
        "by_class": by_class,
        "sc_ss_count": sc_ss,
        "high_rigor": high_rigor,
        "open_actions": open_actions,
        "overdue_actions": overdue_actions,
        "open_reminders": open_reminders,
        "overdue_reminders": overdue_reminders,
        "total_requirements": total_reqs,
        "req_verified_pct": req_verified_pct,
        "reviews_held": reviews_held,
        "reviews_pending": reviews_pending,
        "paperwork_gaps": len(paperwork_due),
    }


def project_analytics() -> dict:
    projects = Project.query.all()
    by_health = OrderedDict()
    by_discipline = OrderedDict()
    for p in projects:
        if p.status in {"planning", "active", "on_hold"}:
            by_health[p.health] = by_health.get(p.health, 0) + 1
            by_discipline[p.discipline] = by_discipline.get(p.discipline, 0) + 1
    return {"by_health": by_health, "by_discipline": by_discipline}
