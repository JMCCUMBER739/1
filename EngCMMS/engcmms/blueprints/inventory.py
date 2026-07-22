"""Spare parts / inventory management."""

from __future__ import annotations

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import login_required

from ..extensions import db
from ..models import Part
from ..permissions import editor_required

bp = Blueprint("inventory", __name__, url_prefix="/inventory")


@bp.route("/")
@login_required
def list_parts():
    show = request.args.get("show", "")
    parts = Part.query.order_by(Part.part_number).all()
    if show == "low":
        parts = [p for p in parts if p.below_reorder]
    total_value = round(sum(p.stock_value for p in Part.query.all()), 2)
    low_count = len([p for p in Part.query.all() if p.below_reorder])
    return render_template("inventory/list.html", parts=parts, show=show,
                           total_value=total_value, low_count=low_count)


@bp.route("/new", methods=["GET", "POST"])
@bp.route("/<int:part_id>/edit", methods=["GET", "POST"])
@login_required
@editor_required
def edit(part_id=None):
    part = db.get_or_404(Part, part_id) if part_id else None
    if request.method == "POST":
        f = request.form
        if part is None:
            part = Part()
            db.session.add(part)
        part.part_number = f.get("part_number", "").strip()
        part.name = f.get("name", "").strip()
        part.description = f.get("description", "")
        part.category = f.get("category", "")
        part.quantity = int(f.get("quantity") or 0)
        part.reorder_point = int(f.get("reorder_point") or 0)
        part.unit_cost = float(f.get("unit_cost") or 0)
        part.location = f.get("location", "")
        part.vendor = f.get("vendor", "")
        if not part.part_number or not part.name:
            flash("Part number and name are required.", "danger")
        else:
            db.session.commit()
            flash("Part saved.", "success")
            return redirect(url_for("inventory.list_parts"))
    return render_template("inventory/edit.html", part=part)


@bp.route("/<int:part_id>/adjust", methods=["POST"])
@login_required
@editor_required
def adjust(part_id):
    part = db.get_or_404(Part, part_id)
    try:
        delta = int(request.form.get("delta", 0))
    except ValueError:
        delta = 0
    part.quantity = max(0, (part.quantity or 0) + delta)
    db.session.commit()
    flash(f"{part.part_number} stock adjusted to {part.quantity}.", "success")
    return redirect(url_for("inventory.list_parts"))
