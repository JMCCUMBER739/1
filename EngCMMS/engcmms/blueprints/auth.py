"""Authentication and account management routes."""

from __future__ import annotations

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from ..extensions import db
from ..models import User

bp = Blueprint("auth", __name__)


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password) and user.is_active:
            login_user(user)
            flash(f"Welcome back, {user.full_name}.", "success")
            next_url = request.args.get("next")
            return redirect(next_url or url_for("dashboard.index"))
        flash("Invalid credentials or inactive account.", "danger")

    return render_template("auth/login.html")


@bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been signed out.", "info")
    return redirect(url_for("auth.login"))


@bp.route("/account", methods=["GET", "POST"])
@login_required
def account():
    if request.method == "POST":
        current_user.full_name = request.form.get("full_name", current_user.full_name)
        current_user.email = request.form.get("email", current_user.email)
        current_user.phone = request.form.get("phone", current_user.phone)
        new_password = request.form.get("new_password", "").strip()
        if new_password:
            current_user.set_password(new_password)
        db.session.commit()
        flash("Account updated.", "success")
        return redirect(url_for("auth.account"))
    return render_template("auth/account.html")
