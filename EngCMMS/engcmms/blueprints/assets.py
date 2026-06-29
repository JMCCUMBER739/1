"""Asset / equipment registry routes."""

from __future__ import annotations

from datetime import datetime

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import login_required

from ..extensions import db
from ..models import (
    ASSET_STATUSES,
    CRITICALITIES,
    DISCIPLINES,
    Asset,
    Document,
    Location,
    WorkOrder,
)
from ..permissions import editor_required

bp = Blueprint("assets", __name__, url_prefix="/assets")


def _parse_date(value):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


@bp.route("/")
@login_required
def list_assets():
    q = request.args.get("q", "").strip()
    status = request.args.get("status", "")
    discipline = request.args.get("discipline", "")

    query = Asset.query
    if q:
        like = f"%{q}%"
        query = query.filter(db.or_(Asset.name.ilike(like), Asset.tag.ilike(like),
                                    Asset.category.ilike(like)))
    if status:
        query = query.filter_by(status=status)
    if discipline:
        query = query.filter_by(discipline=discipline)

    assets = query.order_by(Asset.tag).all()
    return render_template(
        "assets/list.html", assets=assets, statuses=ASSET_STATUSES,
        disciplines=DISCIPLINES, q=q, status=status, discipline=discipline,
    )


@bp.route("/<int:asset_id>")
@login_required
def view(asset_id):
    asset = db.get_or_404(Asset, asset_id)
    workorders = (
        WorkOrder.query.filter_by(asset_id=asset.id)
        .order_by(WorkOrder.created_at.desc()).all()
    )
    docs = Document.query.filter_by(asset_id=asset.id).all()
    return render_template("assets/view.html", asset=asset, workorders=workorders, docs=docs)


@bp.route("/new", methods=["GET", "POST"])
@bp.route("/<int:asset_id>/edit", methods=["GET", "POST"])
@login_required
@editor_required
def edit(asset_id=None):
    asset = db.get_or_404(Asset, asset_id) if asset_id else None

    if request.method == "POST":
        f = request.form
        if asset is None:
            asset = Asset()
            db.session.add(asset)
        asset.tag = f.get("tag", "").strip()
        asset.name = f.get("name", "").strip()
        asset.category = f.get("category", "").strip()
        asset.discipline = f.get("discipline", "other")
        asset.manufacturer = f.get("manufacturer", "")
        asset.model = f.get("model", "")
        asset.serial_number = f.get("serial_number", "")
        asset.status = f.get("status", "operational")
        asset.criticality = f.get("criticality", "medium")
        asset.location_id = int(f["location_id"]) if f.get("location_id") else None
        asset.parent_id = int(f["parent_id"]) if f.get("parent_id") else None
        asset.install_date = _parse_date(f.get("install_date"))
        asset.purchase_cost = float(f.get("purchase_cost") or 0)
        asset.requires_calibration = f.get("requires_calibration") == "on"
        asset.calibration_interval_days = (
            int(f["calibration_interval_days"]) if f.get("calibration_interval_days") else None
        )
        asset.last_calibrated = _parse_date(f.get("last_calibrated"))
        asset.notes = f.get("notes", "")

        if not asset.tag or not asset.name:
            flash("Tag and name are required.", "danger")
        else:
            db.session.commit()
            flash("Asset saved.", "success")
            return redirect(url_for("assets.view", asset_id=asset.id))

    return render_template(
        "assets/edit.html", asset=asset,
        locations=Location.query.order_by(Location.name).all(),
        all_assets=Asset.query.order_by(Asset.tag).all(),
        statuses=ASSET_STATUSES, criticalities=CRITICALITIES, disciplines=DISCIPLINES,
    )


@bp.route("/<int:asset_id>/calibrate", methods=["POST"])
@login_required
@editor_required
def calibrate(asset_id):
    asset = db.get_or_404(Asset, asset_id)
    asset.last_calibrated = _parse_date(request.form.get("date")) or datetime.utcnow().date()
    db.session.commit()
    flash(f"Calibration date updated for {asset.tag}.", "success")
    return redirect(url_for("assets.view", asset_id=asset.id))


@bp.route("/locations", methods=["GET", "POST"])
@login_required
@editor_required
def locations():
    if request.method == "POST":
        loc = Location(
            name=request.form.get("name", "").strip(),
            building=request.form.get("building", ""),
            description=request.form.get("description", ""),
        )
        if loc.name:
            db.session.add(loc)
            db.session.commit()
            flash("Location added.", "success")
        return redirect(url_for("assets.locations"))
    return render_template("assets/locations.html",
                           locations=Location.query.order_by(Location.name).all())
