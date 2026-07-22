"""Database models for the combined CMMS + engineering data tracker."""

from __future__ import annotations

from datetime import datetime, timedelta

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from .extensions import db

# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------
ROLES = ["admin", "engineer", "technician", "viewer"]
DISCIPLINES = ["electrical", "mechanical", "optical", "software", "operations", "other"]

WO_TYPES = ["corrective", "preventive", "calibration", "inspection", "project"]
WO_PRIORITIES = ["critical", "high", "medium", "low"]
WO_STATUSES = ["requested", "open", "in_progress", "on_hold", "completed", "cancelled"]
OPEN_WO_STATUSES = ["requested", "open", "in_progress", "on_hold"]

ASSET_STATUSES = ["operational", "degraded", "down", "maintenance", "retired"]
CRITICALITIES = ["critical", "high", "medium", "low"]

REQUEST_STATUSES = ["new", "triaged", "converted", "rejected"]

PROJECT_STATUSES = ["planning", "active", "on_hold", "completed", "cancelled"]
PROJECT_HEALTH = ["on_track", "at_risk", "off_track"]
PROJECT_TYPES = ["rnd", "characterization", "build", "upgrade", "study"]

DOC_CATEGORIES = ["procedure", "form", "manual", "drawing", "report", "policy", "other"]

# ---------------------------------------------------------------------------
# Design-project configuration-management progress tracker
# (models the FY design-project progress metrics spreadsheet)
# ---------------------------------------------------------------------------
# Maturity code applied to each design-lifecycle stage.
DESIGN_STAGE_CODES = {
    0: "Not Exist",
    1: "Exist / Defined",
    2: "Approved / Checked-In",
    3: "Released",
    4: "Revised",
    5: "N/A",
}
# Ordered lifecycle stages that carry a maturity code.
DESIGN_STAGES = [
    ("req", "REQ", "Requirements"),
    ("ip", "IP", "Implementation Plan"),
    ("cdr", "CDR", "Conceptual Design Review"),
    ("dr", "DR", "Design Reviews"),
    ("dwg", "DWG", "Drawings"),
    ("tpr", "TPR", "Test Plan Results"),
    ("audit", "AUDIT", "Configuration Audit"),
]
DESIGN_PROJECT_STATUSES = ["A", "I", "D", "H"]  # Active, Inactive, Duplicate, Hold
DESIGN_PROJECT_STATUS_LABELS = {"A": "Active", "I": "Inactive", "D": "Duplicate", "H": "Hold"}
DESIGN_TYPES = ["D/A", "P/O", "D/A P/O"]  # Design/Assembly, Process/Operations
DOC_LOCATIONS = {
    "": "—",
    "S": "S:\\Engineering",
    "eP": "ePDM",
    "WC": "WindChill",
    "B": "ePDM & WindChill",
}
# Weight each stage contributes to the maturity roll-up (N/A stages excluded).
_STAGE_PROGRESS = {0: 0.0, 1: 0.34, 2: 0.67, 3: 1.0, 4: 1.0}

# ---------------------------------------------------------------------------
# Email-to-server intake: document submissions that map onto design stages.
# (code, human label, DesignProject stage field it advances)
# ---------------------------------------------------------------------------
SUBMISSION_TYPES = [
    ("REQ", "Requirements Document", "req"),
    ("IP", "Implementation Plan", "ip"),
    ("CDR", "Conceptual Design Review", "cdr"),
    ("DR", "Design Review", "dr"),
    ("DWG", "Drawings", "dwg"),
    ("TPR", "Test Plan Results", "tpr"),
    ("AUDIT", "Configuration Audit", "audit"),
    ("OTHER", "Other / General", None),
]
SUBMISSION_TYPE_FIELD = {code: field for code, _label, field in SUBMISSION_TYPES}
SUBMISSION_TYPE_LABEL = {code: label for code, label, _field in SUBMISSION_TYPES}
SUBMISSION_STATUSES = ["received", "applied", "rejected"]

# Office/Word/ODF template file types allowed for the team template library.
TEMPLATE_EXTENSIONS = {"docx", "doc", "odt", "odf", "pdf", "rtf", "txt", "xlsx", "pptx"}


# ---------------------------------------------------------------------------
# Users & permissions
# ---------------------------------------------------------------------------
class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    full_name = db.Column(db.String(160), nullable=False)
    email = db.Column(db.String(200), unique=True, nullable=False)
    role = db.Column(db.String(20), nullable=False, default="viewer")
    discipline = db.Column(db.String(40), default="other")
    title = db.Column(db.String(120))
    phone = db.Column(db.String(40))
    is_active_user = db.Column(db.Boolean, default=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    assigned_workorders = db.relationship(
        "WorkOrder", back_populates="assignee", foreign_keys="WorkOrder.assignee_id"
    )

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    @property
    def is_active(self) -> bool:  # used by Flask-Login
        return bool(self.is_active_user)

    def can(self, *roles: str) -> bool:
        return self.role in roles

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def can_edit(self) -> bool:
        """Admins, engineers and technicians may create / edit records."""
        return self.role in {"admin", "engineer", "technician"}

    def __repr__(self) -> str:  # pragma: no cover
        return f"<User {self.username} ({self.role})>"


# ---------------------------------------------------------------------------
# Locations & assets
# ---------------------------------------------------------------------------
class Location(db.Model):
    __tablename__ = "locations"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(160), nullable=False)
    building = db.Column(db.String(120))
    description = db.Column(db.Text)

    assets = db.relationship("Asset", back_populates="location")


class Asset(db.Model):
    """A piece of equipment, instrument or laboratory system."""

    __tablename__ = "assets"

    id = db.Column(db.Integer, primary_key=True)
    tag = db.Column(db.String(60), unique=True, nullable=False)
    name = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(80))  # e.g. laser, oscilloscope, vacuum pump
    discipline = db.Column(db.String(40), default="other")
    manufacturer = db.Column(db.String(120))
    model = db.Column(db.String(120))
    serial_number = db.Column(db.String(120))
    status = db.Column(db.String(20), default="operational")
    criticality = db.Column(db.String(20), default="medium")
    location_id = db.Column(db.Integer, db.ForeignKey("locations.id"))
    parent_id = db.Column(db.Integer, db.ForeignKey("assets.id"))

    install_date = db.Column(db.Date)
    purchase_cost = db.Column(db.Float, default=0.0)

    # Calibration tracking (key for diagnostics labs).
    requires_calibration = db.Column(db.Boolean, default=False)
    calibration_interval_days = db.Column(db.Integer)
    last_calibrated = db.Column(db.Date)

    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    location = db.relationship("Location", back_populates="assets")
    children = db.relationship("Asset", backref=db.backref("parent", remote_side=[id]))
    workorders = db.relationship("WorkOrder", back_populates="asset")
    pm_schedules = db.relationship("PMSchedule", back_populates="asset")

    @property
    def next_calibration_due(self):
        if not (self.requires_calibration and self.last_calibrated and self.calibration_interval_days):
            return None
        return self.last_calibrated + timedelta(days=self.calibration_interval_days)

    @property
    def calibration_overdue(self) -> bool:
        due = self.next_calibration_due
        return bool(due and due < datetime.utcnow().date())


# ---------------------------------------------------------------------------
# Work orders & maintenance requests
# ---------------------------------------------------------------------------
class WorkOrder(db.Model):
    __tablename__ = "workorders"

    id = db.Column(db.Integer, primary_key=True)
    number = db.Column(db.String(30), unique=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    wo_type = db.Column(db.String(20), default="corrective")
    priority = db.Column(db.String(20), default="medium")
    status = db.Column(db.String(20), default="open")

    asset_id = db.Column(db.Integer, db.ForeignKey("assets.id"))
    assignee_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    pm_schedule_id = db.Column(db.Integer, db.ForeignKey("pm_schedules.id"))
    requested_by = db.Column(db.String(160))

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    due_date = db.Column(db.DateTime)
    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)

    # Metrics captured at completion.
    labor_hours = db.Column(db.Float, default=0.0)
    downtime_hours = db.Column(db.Float, default=0.0)
    cost_parts = db.Column(db.Float, default=0.0)
    cost_labor = db.Column(db.Float, default=0.0)
    failure_code = db.Column(db.String(120))
    resolution = db.Column(db.Text)

    asset = db.relationship("Asset", back_populates="workorders")
    assignee = db.relationship(
        "User", back_populates="assigned_workorders", foreign_keys=[assignee_id]
    )
    pm_schedule = db.relationship("PMSchedule", back_populates="workorders")
    attachments = db.relationship(
        "Document", back_populates="workorder", cascade="all, delete-orphan"
    )

    @property
    def total_cost(self) -> float:
        return (self.cost_parts or 0) + (self.cost_labor or 0)

    @property
    def is_open(self) -> bool:
        return self.status in OPEN_WO_STATUSES

    @property
    def is_overdue(self) -> bool:
        return bool(self.is_open and self.due_date and self.due_date < datetime.utcnow())

    @property
    def resolution_hours(self):
        if self.completed_at and self.created_at:
            return round((self.completed_at - self.created_at).total_seconds() / 3600.0, 2)
        return None


class MaintenanceRequest(db.Model):
    """Inbound request, either entered manually or parsed from an email."""

    __tablename__ = "requests"

    id = db.Column(db.Integer, primary_key=True)
    subject = db.Column(db.String(200), nullable=False)
    body = db.Column(db.Text)
    requester_name = db.Column(db.String(160))
    requester_email = db.Column(db.String(200))
    source = db.Column(db.String(20), default="manual")  # manual | email
    priority = db.Column(db.String(20), default="medium")
    status = db.Column(db.String(20), default="new")
    asset_id = db.Column(db.Integer, db.ForeignKey("assets.id"))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    workorder_id = db.Column(db.Integer, db.ForeignKey("workorders.id"))

    asset = db.relationship("Asset")
    workorder = db.relationship("WorkOrder")


# ---------------------------------------------------------------------------
# Preventive maintenance
# ---------------------------------------------------------------------------
class PMSchedule(db.Model):
    __tablename__ = "pm_schedules"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    asset_id = db.Column(db.Integer, db.ForeignKey("assets.id"))
    assignee_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    frequency_days = db.Column(db.Integer, default=30)
    priority = db.Column(db.String(20), default="medium")
    estimated_hours = db.Column(db.Float, default=1.0)
    last_generated = db.Column(db.Date)
    next_due = db.Column(db.Date)
    active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    asset = db.relationship("Asset", back_populates="pm_schedules")
    assignee = db.relationship("User")
    workorders = db.relationship("WorkOrder", back_populates="pm_schedule")

    @property
    def is_due(self) -> bool:
        return bool(self.active and self.next_due and self.next_due <= datetime.utcnow().date())


# ---------------------------------------------------------------------------
# Inventory / spare parts
# ---------------------------------------------------------------------------
class Part(db.Model):
    __tablename__ = "parts"

    id = db.Column(db.Integer, primary_key=True)
    part_number = db.Column(db.String(80), unique=True, nullable=False)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    category = db.Column(db.String(80))
    quantity = db.Column(db.Integer, default=0)
    reorder_point = db.Column(db.Integer, default=0)
    unit_cost = db.Column(db.Float, default=0.0)
    location = db.Column(db.String(120))
    vendor = db.Column(db.String(160))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    @property
    def below_reorder(self) -> bool:
        return self.quantity <= self.reorder_point

    @property
    def stock_value(self) -> float:
        return (self.quantity or 0) * (self.unit_cost or 0)


# ---------------------------------------------------------------------------
# Document / forms repository
# ---------------------------------------------------------------------------
class Document(db.Model):
    __tablename__ = "documents"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(40), default="procedure")
    description = db.Column(db.Text)
    revision = db.Column(db.String(40))
    stored_filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    content_type = db.Column(db.String(120))
    size_bytes = db.Column(db.Integer, default=0)
    uploaded_by_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    workorder_id = db.Column(db.Integer, db.ForeignKey("workorders.id"))
    asset_id = db.Column(db.Integer, db.ForeignKey("assets.id"))
    project_id = db.Column(db.Integer, db.ForeignKey("projects.id"))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    uploaded_by = db.relationship("User")
    workorder = db.relationship("WorkOrder", back_populates="attachments")
    asset = db.relationship("Asset")
    project = db.relationship("Project", back_populates="attachments")

    @property
    def size_kb(self) -> float:
        return round((self.size_bytes or 0) / 1024.0, 1)


# ---------------------------------------------------------------------------
# Engineering projects & data tracking
# ---------------------------------------------------------------------------
class Project(db.Model):
    __tablename__ = "projects"

    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(40), unique=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    project_type = db.Column(db.String(30), default="rnd")
    discipline = db.Column(db.String(40), default="other")
    status = db.Column(db.String(20), default="active")
    health = db.Column(db.String(20), default="on_track")
    lead_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    percent_complete = db.Column(db.Integer, default=0)
    budget = db.Column(db.Float, default=0.0)
    spent = db.Column(db.Float, default=0.0)
    start_date = db.Column(db.Date)
    target_date = db.Column(db.Date)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    lead = db.relationship("User")
    updates = db.relationship(
        "ProjectUpdate", back_populates="project",
        cascade="all, delete-orphan", order_by="ProjectUpdate.created_at.desc()",
    )
    attachments = db.relationship("Document", back_populates="project")

    @property
    def is_overdue(self) -> bool:
        active = self.status in {"planning", "active", "on_hold"}
        return bool(active and self.target_date and self.target_date < datetime.utcnow().date())

    @property
    def budget_used_pct(self):
        if not self.budget:
            return None
        return round((self.spent or 0) / self.budget * 100.0, 1)


class ProjectUpdate(db.Model):
    """A logged measurement, milestone, or status note for a project."""

    __tablename__ = "project_updates"

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey("projects.id"), nullable=False)
    author_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    update_type = db.Column(db.String(30), default="note")  # note | milestone | metric | risk
    title = db.Column(db.String(200))
    body = db.Column(db.Text)
    metric_name = db.Column(db.String(120))
    metric_value = db.Column(db.Float)
    metric_unit = db.Column(db.String(40))
    percent_complete = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    project = db.relationship("Project", back_populates="updates")
    author = db.relationship("User")


class DesignProject(db.Model):
    """A design project tracked through its configuration-management lifecycle.

    Mirrors the FY design-project progress-metrics spreadsheet: each project
    carries document identifiers, responsible roles, an active/inactive status,
    and a maturity code (0-5) for each lifecycle stage.
    """

    __tablename__ = "design_projects"

    id = db.Column(db.Integer, primary_key=True)
    number = db.Column(db.Integer)  # display / sheet row number
    title = db.Column(db.String(200), nullable=False)
    design_name = db.Column(db.String(200))

    # Document / PLM identifiers
    windchill_number = db.Column(db.String(80))
    epdm_number = db.Column(db.String(80))
    program = db.Column(db.String(80))  # e.g. SEO, LLNL, LANL, LAO, HEDE/S&T
    doc_location = db.Column(db.String(4), default="")  # S | eP | WC | B

    # Responsible roles (free text to allow external / partner names)
    design_authority = db.Column(db.String(120))
    pm = db.Column(db.String(120))  # project manager
    dm = db.Column(db.String(120))  # design manager
    dtl = db.Column(db.String(120))  # design team lead

    da_po = db.Column(db.String(12), default="D/A")  # D/A | P/O | D/A P/O
    status = db.Column(db.String(2), default="A")  # A | I | D | H

    # Lifecycle-stage maturity codes (0-5)
    req = db.Column(db.Integer, default=0)
    ip = db.Column(db.Integer, default=0)
    cdr = db.Column(db.Integer, default=0)
    dr = db.Column(db.Integer, default=0)
    dwg = db.Column(db.Integer, default=0)
    tpr = db.Column(db.Integer, default=0)
    audit = db.Column(db.Integer, default=0)

    # Acceptance / approval
    accepted = db.Column(db.String(1), default="")  # Y | N | ""
    accepted_by = db.Column(db.String(40))  # reviewer initials

    # Notification tracking
    email_sent = db.Column(db.Boolean, default=False)
    email_sent_at = db.Column(db.DateTime)

    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    STAGE_FIELDS = ["req", "ip", "cdr", "dr", "dwg", "tpr", "audit"]

    @property
    def is_active(self) -> bool:
        return self.status == "A"

    @property
    def maturity_pct(self) -> float:
        """Roll-up completion across all non-N/A lifecycle stages."""
        weights = []
        for field in self.STAGE_FIELDS:
            code = getattr(self, field)
            if code is None or code == 5:  # 5 = N/A -> excluded
                continue
            weights.append(_STAGE_PROGRESS.get(code, 0.0))
        if not weights:
            return 0.0
        return round(sum(weights) / len(weights) * 100, 0)

    @property
    def accept_display(self) -> str:
        if not self.accepted:
            return "—"
        label = self.accepted
        if self.number:
            label += str(self.number)
        if self.accepted_by:
            label += f" - {self.accepted_by}"
        return label


# ---------------------------------------------------------------------------
# Notifications: contacts, email templates, sent-email log
# ---------------------------------------------------------------------------
class Contact(db.Model):
    """A notification recipient with subscription preferences."""

    __tablename__ = "contacts"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(160), nullable=False)
    email = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(120))
    discipline = db.Column(db.String(40), default="other")
    active = db.Column(db.Boolean, default=True)

    # Subscription toggles drive which auto-generated emails a contact receives.
    notify_workorders = db.Column(db.Boolean, default=True)
    notify_pm = db.Column(db.Boolean, default=True)
    notify_calibration = db.Column(db.Boolean, default=True)
    notify_projects = db.Column(db.Boolean, default=True)
    notify_digest = db.Column(db.Boolean, default=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class EmailTemplate(db.Model):
    __tablename__ = "email_templates"

    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(60), unique=True)
    name = db.Column(db.String(160), nullable=False)
    category = db.Column(db.String(60), default="general")
    subject = db.Column(db.String(255), nullable=False)
    body = db.Column(db.Text, nullable=False)
    description = db.Column(db.Text)
    is_builtin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class EmailLog(db.Model):
    __tablename__ = "email_log"

    id = db.Column(db.Integer, primary_key=True)
    subject = db.Column(db.String(255))
    recipients = db.Column(db.Text)
    body = db.Column(db.Text)
    category = db.Column(db.String(60))
    attachments = db.Column(db.Text)  # comma-separated attachment filenames
    status = db.Column(db.String(20), default="preview")  # preview | sent | failed
    error = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class MessageTemplate(db.Model):
    """An uploaded Word / ODF / PDF document used as a reusable team message.

    Supports placeholder substitution in ``.docx`` files (tokens like
    ``{{ name }}``) when python-docx is installed; otherwise the file is sent
    as-is.  Ideal for standard forms, memos and status templates.
    """

    __tablename__ = "message_templates"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    category = db.Column(db.String(60), default="general")
    stored_filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    content_type = db.Column(db.String(120))
    size_bytes = db.Column(db.Integer, default=0)
    uploaded_by_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    uploaded_by = db.relationship("User")

    @property
    def extension(self) -> str:
        return self.original_filename.rsplit(".", 1)[-1].lower() if "." in self.original_filename else ""

    @property
    def size_kb(self) -> float:
        return round((self.size_bytes or 0) / 1024.0, 1)


class Submission(db.Model):
    """An inbound document submission (e.g. an emailed IP or REQ).

    Submissions can be logged manually, or posted to the intake API by a mail
    gateway.  Applying a submission advances the matching stage on its linked
    design project and attaches the file to that project.
    """

    __tablename__ = "submissions"

    id = db.Column(db.Integer, primary_key=True)
    submission_type = db.Column(db.String(12), default="OTHER")  # REQ, IP, ...
    design_project_id = db.Column(db.Integer, db.ForeignKey("design_projects.id"))
    project_ref = db.Column(db.String(120))  # free-text hint (code/title) if unmatched
    sender_name = db.Column(db.String(160))
    sender_email = db.Column(db.String(200))
    subject = db.Column(db.String(255))
    body = db.Column(db.Text)
    document_id = db.Column(db.Integer, db.ForeignKey("documents.id"))
    resulting_code = db.Column(db.Integer, default=2)  # stage code to set on apply
    source = db.Column(db.String(20), default="manual")  # manual | email | api
    status = db.Column(db.String(20), default="received")  # received | applied | rejected
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    applied_at = db.Column(db.DateTime)

    design_project = db.relationship("DesignProject")
    document = db.relationship("Document")
