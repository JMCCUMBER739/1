"""Notification contacts with auto-generated email addresses."""

from __future__ import annotations

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import login_required

from .. import email_service
from ..extensions import db
from ..models import DISCIPLINES, Contact
from ..permissions import editor_required

bp = Blueprint("contacts", __name__, url_prefix="/contacts")


@bp.route("/")
@login_required
def list_contacts():
    contacts = Contact.query.order_by(Contact.name).all()
    return render_template("contacts/list.html", contacts=contacts,
                           disciplines=DISCIPLINES)


@bp.route("/suggest-email")
@login_required
def suggest_email():
    """Live helper used by the contact form to auto-generate an address."""
    name = request.args.get("name", "")
    return {"email": email_service.generate_email_address(name)}


@bp.route("/new", methods=["POST"])
@bp.route("/<int:contact_id>/edit", methods=["POST"])
@login_required
@editor_required
def save(contact_id=None):
    contact = db.get_or_404(Contact, contact_id) if contact_id else None
    f = request.form
    if contact is None:
        contact = Contact()
        db.session.add(contact)
    contact.name = f.get("name", "").strip()
    contact.email = f.get("email", "").strip()
    # Auto-generate an address from the name when none was supplied.
    if contact.name and not contact.email:
        contact.email = email_service.generate_email_address(contact.name)
    contact.role = f.get("role", "")
    contact.discipline = f.get("discipline", "other")
    contact.active = f.get("active") == "on"
    contact.notify_workorders = f.get("notify_workorders") == "on"
    contact.notify_pm = f.get("notify_pm") == "on"
    contact.notify_calibration = f.get("notify_calibration") == "on"
    contact.notify_projects = f.get("notify_projects") == "on"
    contact.notify_digest = f.get("notify_digest") == "on"

    if not contact.name:
        flash("Name is required.", "danger")
    else:
        db.session.commit()
        flash("Contact saved.", "success")
    return redirect(url_for("contacts.list_contacts"))


@bp.route("/<int:contact_id>/delete", methods=["POST"])
@login_required
@editor_required
def delete(contact_id):
    contact = db.get_or_404(Contact, contact_id)
    db.session.delete(contact)
    db.session.commit()
    flash("Contact removed.", "info")
    return redirect(url_for("contacts.list_contacts"))
