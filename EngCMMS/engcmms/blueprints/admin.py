"""Administration: user management and system settings."""

from __future__ import annotations

from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import login_required

from .. import email_service
from ..extensions import db
from ..models import DISCIPLINES, ROLES, User
from ..permissions import admin_required

bp = Blueprint("admin", __name__, url_prefix="/admin")


@bp.route("/users")
@login_required
@admin_required
def users():
    return render_template("admin/users.html",
                           users=User.query.order_by(User.full_name).all(),
                           roles=ROLES, disciplines=DISCIPLINES)


@bp.route("/users/new", methods=["POST"])
@bp.route("/users/<int:user_id>/edit", methods=["POST"])
@login_required
@admin_required
def save_user(user_id=None):
    user = db.get_or_404(User, user_id) if user_id else None
    f = request.form
    if user is None:
        user = User()
        db.session.add(user)
        user.set_password(f.get("password") or "changeme")
    user.username = f.get("username", "").strip()
    user.full_name = f.get("full_name", "").strip()
    user.email = f.get("email", "").strip() or email_service.generate_email_address(user.full_name)
    user.role = f.get("role", "viewer")
    user.discipline = f.get("discipline", "other")
    user.title = f.get("title", "")
    user.phone = f.get("phone", "")
    user.is_active_user = f.get("is_active_user") == "on"
    if f.get("password"):
        user.set_password(f["password"])

    if not user.username or not user.full_name:
        flash("Username and full name are required.", "danger")
    else:
        try:
            db.session.commit()
            flash("User saved.", "success")
        except Exception:
            db.session.rollback()
            flash("Could not save user (duplicate username/email?).", "danger")
    return redirect(url_for("admin.users"))


@bp.route("/settings")
@login_required
@admin_required
def settings():
    cfg = current_app.config
    info = {
        "Data directory": str(cfg.get("DATA_DIR")),
        "Upload directory": str(cfg.get("UPLOAD_DIR")),
        "Database": cfg.get("SQLALCHEMY_DATABASE_URI"),
        "Mail server": cfg.get("MAIL_SERVER") or "(preview mode — no SMTP configured)",
        "Mail sender": cfg.get("MAIL_DEFAULT_SENDER"),
        "Email domain": cfg.get("EMAIL_DOMAIN"),
        "Organization": cfg.get("ORG_NAME"),
        "Max upload (MB)": cfg.get("MAX_CONTENT_LENGTH", 0) // (1024 * 1024),
    }
    return render_template("admin/settings.html", info=info)
