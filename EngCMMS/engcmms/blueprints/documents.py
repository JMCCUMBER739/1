"""Document & forms repository with file attachments."""

from __future__ import annotations

import os
import uuid

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename

from ..extensions import db
from ..models import (
    DOC_CATEGORIES,
    Asset,
    Document,
    Project,
    WorkOrder,
)
from ..permissions import editor_required

bp = Blueprint("documents", __name__, url_prefix="/documents")


def _allowed(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in current_app.config["ALLOWED_UPLOAD_EXTENSIONS"]


@bp.route("/")
@login_required
def list_documents():
    category = request.args.get("category", "")
    q = request.args.get("q", "").strip()
    query = Document.query
    if category:
        query = query.filter_by(category=category)
    if q:
        query = query.filter(Document.title.ilike(f"%{q}%"))
    docs = query.order_by(Document.created_at.desc()).all()
    return render_template("documents/list.html", docs=docs,
                           categories=DOC_CATEGORIES, category=category, q=q)


@bp.route("/upload", methods=["GET", "POST"])
@login_required
@editor_required
def upload():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("Please choose a file.", "danger")
            return redirect(url_for("documents.upload"))
        if not _allowed(file.filename):
            flash("File type not allowed.", "danger")
            return redirect(url_for("documents.upload"))

        original = secure_filename(file.filename)
        stored = f"{uuid.uuid4().hex}_{original}"
        path = os.path.join(current_app.config["UPLOAD_DIR"], stored)
        file.save(path)

        f = request.form
        doc = Document(
            title=f.get("title", "").strip() or original,
            category=f.get("category", "procedure"),
            description=f.get("description", ""),
            revision=f.get("revision", ""),
            stored_filename=stored,
            original_filename=original,
            content_type=file.mimetype,
            size_bytes=os.path.getsize(path),
            uploaded_by_id=current_user.id,
            asset_id=int(f["asset_id"]) if f.get("asset_id") else None,
            workorder_id=int(f["workorder_id"]) if f.get("workorder_id") else None,
            project_id=int(f["project_id"]) if f.get("project_id") else None,
        )
        db.session.add(doc)
        db.session.commit()
        flash("Document uploaded.", "success")
        return redirect(url_for("documents.list_documents"))

    return render_template(
        "documents/upload.html", categories=DOC_CATEGORIES,
        assets=Asset.query.order_by(Asset.tag).all(),
        workorders=WorkOrder.query.order_by(WorkOrder.id.desc()).limit(100).all(),
        projects=Project.query.order_by(Project.code).all(),
        preselect_workorder=request.args.get("workorder", type=int),
        preselect_asset=request.args.get("asset", type=int),
        preselect_project=request.args.get("project", type=int),
    )


@bp.route("/<int:doc_id>/download")
@login_required
def download(doc_id):
    doc = db.get_or_404(Document, doc_id)
    directory = current_app.config["UPLOAD_DIR"]
    if not os.path.exists(os.path.join(directory, doc.stored_filename)):
        abort(404)
    return send_from_directory(
        directory, doc.stored_filename, as_attachment=True,
        download_name=doc.original_filename,
    )


@bp.route("/<int:doc_id>/delete", methods=["POST"])
@login_required
@editor_required
def delete(doc_id):
    doc = db.get_or_404(Document, doc_id)
    path = os.path.join(current_app.config["UPLOAD_DIR"], doc.stored_filename)
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
    db.session.delete(doc)
    db.session.commit()
    flash("Document deleted.", "info")
    return redirect(url_for("documents.list_documents"))
