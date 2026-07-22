"""EngCMMS application factory.

A combined Computerized Maintenance Management System (CMMS) and engineering
data tracker for laboratory diagnostics teams.
"""

from __future__ import annotations

from datetime import datetime

from flask import Flask

from .config import get_config
from .extensions import db, login_manager

__version__ = "1.0.0"


def create_app(config_object=None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_object or get_config())

    db.init_app(app)
    login_manager.init_app(app)

    from .models import User

    @login_manager.user_loader
    def load_user(user_id):  # pragma: no cover - trivial
        return db.session.get(User, int(user_id))

    _register_blueprints(app)
    _register_context(app)
    _register_errors(app)

    with app.app_context():
        db.create_all()
        from .seed import ensure_bootstrap

        ensure_bootstrap()

    return app


def _register_blueprints(app: Flask) -> None:
    from .blueprints import (
        admin,
        assets,
        auth,
        contacts,
        dashboard,
        design,
        documents,
        emails,
        inventory,
        pm,
        projects,
        requests_bp,
        workorders,
    )

    app.register_blueprint(auth.bp)
    app.register_blueprint(dashboard.bp)
    app.register_blueprint(assets.bp)
    app.register_blueprint(workorders.bp)
    app.register_blueprint(pm.bp)
    app.register_blueprint(inventory.bp)
    app.register_blueprint(requests_bp.bp)
    app.register_blueprint(documents.bp)
    app.register_blueprint(projects.bp)
    app.register_blueprint(design.bp)
    app.register_blueprint(contacts.bp)
    app.register_blueprint(emails.bp)
    app.register_blueprint(admin.bp)


def _register_context(app: Flask) -> None:
    from .models import (
        MaintenanceRequest,
        WorkOrder,
        OPEN_WO_STATUSES,
    )

    @app.context_processor
    def inject_globals():
        from flask_login import current_user

        pending_requests = 0
        open_wos = 0
        if current_user.is_authenticated:
            pending_requests = MaintenanceRequest.query.filter_by(status="new").count()
            open_wos = WorkOrder.query.filter(
                WorkOrder.status.in_(OPEN_WO_STATUSES)
            ).count()
        return {
            "app_version": __version__,
            "org_name": app.config.get("ORG_NAME"),
            "now": datetime.utcnow(),
            "nav_pending_requests": pending_requests,
            "nav_open_workorders": open_wos,
        }

    @app.template_filter("dt")
    def _format_dt(value, fmt="%Y-%m-%d %H:%M"):
        if not value:
            return "—"
        return value.strftime(fmt)

    @app.template_filter("d")
    def _format_d(value, fmt="%Y-%m-%d"):
        if not value:
            return "—"
        return value.strftime(fmt)

    @app.template_filter("money")
    def _money(value):
        try:
            return f"${value:,.2f}"
        except (TypeError, ValueError):
            return "$0.00"


def _register_errors(app: Flask) -> None:
    from flask import render_template

    messages = {
        401: "You need to sign in to view this page.",
        403: "You do not have permission to perform this action.",
        404: "The page or record you requested was not found.",
        413: "That upload exceeds the maximum allowed size.",
        500: "An unexpected error occurred.",
    }

    def handler(error):
        code = getattr(error, "code", 500)
        return render_template("error.html", code=code,
                               message=messages.get(code, "Something went wrong.")), code

    for code in (401, 403, 404, 413, 500):
        app.register_error_handler(code, handler)
