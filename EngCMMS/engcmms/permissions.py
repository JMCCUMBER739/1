"""Role-based access control decorators."""

from __future__ import annotations

from functools import wraps

from flask import abort
from flask_login import current_user


def roles_required(*roles: str):
    """Allow the view only for the listed roles."""

    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if not current_user.is_authenticated:
                abort(401)
            if current_user.role not in roles:
                abort(403)
            return view(*args, **kwargs)

        return wrapped

    return decorator


def editor_required(view):
    """Allow admins, engineers and technicians (anyone who can edit records)."""

    @wraps(view)
    def wrapped(*args, **kwargs):
        if not current_user.is_authenticated:
            abort(401)
        if not current_user.can_edit:
            abort(403)
        return view(*args, **kwargs)

    return wrapped


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not current_user.is_authenticated:
            abort(401)
        if not current_user.is_admin:
            abort(403)
        return view(*args, **kwargs)

    return wrapped
