"""Bootstrap defaults and optional rich demo data.

``ensure_bootstrap`` runs on every startup and is idempotent: it guarantees an
admin account and the built-in email templates exist.  ``seed_demo`` populates a
realistic sample data set for evaluation and screenshots.
"""

from __future__ import annotations

import os
import random
from datetime import datetime, timedelta

from .extensions import db
from .models import (
    Asset,
    Contact,
    DesignProject,
    EmailTemplate,
    Location,
    PMSchedule,
    Part,
    Project,
    ProjectUpdate,
    User,
    WorkOrder,
)

BUILTIN_TEMPLATES = [
    {
        "key": "wo_assigned",
        "name": "Work Order Assigned",
        "category": "workorder",
        "subject": "[{{ wo_priority }}] Work Order {{ wo_number }}: {{ wo_title }}",
        "body": (
            "Hello {{ assignee }},\n\n"
            "Work order {{ wo_number }} ({{ wo_type }}) has been assigned to you.\n\n"
            "Asset: {{ asset_name }}\n"
            "Priority: {{ wo_priority }}\n"
            "Status: {{ wo_status }}\n"
            "Due: {{ due_date }}\n\n"
            "Details:\n{{ description }}\n\n"
            "— {{ org_name }} EngCMMS"
        ),
        "is_builtin": True,
    },
    {
        "key": "weekly_update",
        "name": "Weekly Team Update",
        "category": "digest",
        "subject": "{{ org_name }} — Weekly Engineering & Maintenance Update",
        "body": (
            "Team,\n\n"
            "Here is the weekly status snapshot.\n\n"
            "MAINTENANCE\n"
            "  Open work orders: {{ open_workorders }}\n"
            "  Overdue: {{ overdue_workorders }}\n"
            "  PM compliance: {{ pm_compliance }}%\n"
            "  Asset availability: {{ asset_availability }}%\n"
            "  Calibrations due (14d): {{ calibrations_due }}\n\n"
            "ENGINEERING\n"
            "  Active projects: {{ active_projects }}\n"
            "  Projects at risk: {{ projects_at_risk }}\n\n"
            "Notes:\n{{ notes }}\n\n"
            "— {{ org_name }}"
        ),
        "is_builtin": True,
    },
    {
        "key": "downtime_alert",
        "name": "Equipment Downtime Alert",
        "category": "workorder",
        "subject": "DOWNTIME ALERT: {{ asset_name }}",
        "body": (
            "Team,\n\n"
            "{{ asset_name }} ({{ asset_tag }}) is currently DOWN.\n\n"
            "Impact: {{ impact }}\n"
            "Work order: {{ wo_number }}\n"
            "Assigned to: {{ assignee }}\n"
            "Estimated return to service: {{ eta }}\n\n"
            "Please plan lab activities accordingly.\n\n"
            "— {{ org_name }}"
        ),
        "is_builtin": True,
    },
    {
        "key": "calibration_due",
        "name": "Calibration Due Reminder",
        "category": "calibration",
        "subject": "Calibration due: {{ asset_name }} by {{ due_date }}",
        "body": (
            "Hello,\n\n"
            "The following instrument is due for calibration:\n\n"
            "Instrument: {{ asset_name }} ({{ asset_tag }})\n"
            "Last calibrated: {{ last_calibrated }}\n"
            "Due by: {{ due_date }}\n\n"
            "Please schedule calibration to maintain measurement traceability.\n\n"
            "— {{ org_name }}"
        ),
        "is_builtin": True,
    },
    {
        "key": "project_status",
        "name": "Project Status Update",
        "category": "project",
        "subject": "Project {{ project_code }} — {{ project_name }} ({{ health }})",
        "body": (
            "Team,\n\n"
            "Status update for {{ project_name }} ({{ project_code }}).\n\n"
            "Type: {{ project_type }}\n"
            "Lead: {{ lead }}\n"
            "Health: {{ health }}\n"
            "Progress: {{ percent_complete }}%\n"
            "Target date: {{ target_date }}\n\n"
            "Summary:\n{{ summary }}\n\n"
            "— {{ org_name }}"
        ),
        "is_builtin": True,
    },
    {
        "key": "maintenance_notice",
        "name": "Scheduled Maintenance Notice",
        "category": "pm",
        "subject": "Scheduled maintenance: {{ asset_name }} on {{ date }}",
        "body": (
            "Team,\n\n"
            "Scheduled maintenance is planned for {{ asset_name }} ({{ asset_tag }}).\n\n"
            "Date/Window: {{ date }}\n"
            "Expected downtime: {{ downtime }}\n"
            "Performed by: {{ assignee }}\n\n"
            "Please save data and secure experiments before this window.\n\n"
            "— {{ org_name }}"
        ),
        "is_builtin": True,
    },
]


def ensure_bootstrap() -> None:
    """Create the default admin and built-in templates if they are missing."""
    _ensure_templates()
    if User.query.count() == 0:
        admin = User(
            username="admin",
            full_name="System Administrator",
            email="admin@lab.local",
            role="admin",
            discipline="operations",
            title="Engineering Lead",
        )
        admin.set_password(os.environ.get("ENGCMMS_ADMIN_PASSWORD", "admin"))
        db.session.add(admin)
        db.session.commit()


def _ensure_templates() -> None:
    for tpl in BUILTIN_TEMPLATES:
        if not EmailTemplate.query.filter_by(key=tpl["key"]).first():
            db.session.add(EmailTemplate(**tpl))
    db.session.commit()


# ---------------------------------------------------------------------------
# Rich demo data
# ---------------------------------------------------------------------------
def seed_demo() -> None:
    """Populate a realistic data set.  Safe to run once on an empty database."""
    ensure_bootstrap()
    if Asset.query.count() > 0:
        print("Demo data already present; skipping.")
        return

    random.seed(7)
    now = datetime.utcnow()

    # --- People -----------------------------------------------------------
    people = [
        ("jchen", "Dr. Jordan Chen", "engineer", "optical", "Optical Engineer"),
        ("mrivera", "Maria Rivera", "engineer", "electrical", "Electrical Engineer"),
        ("tsato", "Tom Sato", "engineer", "mechanical", "Mechanical Engineer"),
        ("kpatel", "Kiran Patel", "technician", "operations", "Lab Technician"),
        ("lwagner", "Lena Wagner", "technician", "operations", "Lab Technician"),
        ("viewer", "Program Manager", "viewer", "operations", "Program Manager"),
    ]
    users = []
    for username, name, role, disc, title in people:
        u = User(
            username=username, full_name=name,
            email=f"{username}@lab.doe.local", role=role,
            discipline=disc, title=title,
        )
        u.set_password("password")
        db.session.add(u)
        users.append(u)
    db.session.commit()
    admin = User.query.filter_by(username="admin").first()
    engineers = [u for u in users if u.role in {"engineer", "technician"}]

    # --- Locations --------------------------------------------------------
    locations = [
        Location(name="Optics Lab A", building="Bldg 401", description="Class 1000 cleanroom"),
        Location(name="Pulsed Power Bay", building="Bldg 401", description="High-voltage area"),
        Location(name="Metrology Lab", building="Bldg 402", description="Climate controlled"),
        Location(name="Machine Shop", building="Bldg 403", description="Fabrication"),
    ]
    db.session.add_all(locations)
    db.session.commit()

    # --- Assets -----------------------------------------------------------
    asset_specs = [
        ("LAS-001", "Nd:YAG Pump Laser", "laser", "optical", "critical", True, 180),
        ("OSC-014", "20 GHz Oscilloscope", "oscilloscope", "electrical", "high", True, 365),
        ("SPEC-003", "Imaging Spectrometer", "spectrometer", "optical", "high", True, 365),
        ("VAC-007", "Turbo Vacuum Pump", "vacuum pump", "mechanical", "high", False, None),
        ("PSU-022", "High Voltage Power Supply", "power supply", "electrical", "critical", True, 365),
        ("CHL-002", "Recirculating Chiller", "chiller", "mechanical", "medium", False, None),
        ("CAM-009", "Streak Camera", "camera", "optical", "critical", True, 180),
        ("DAQ-031", "High-Speed Digitizer", "daq", "electrical", "high", True, 365),
        ("MNT-005", "Optical Table", "optical table", "optical", "medium", False, None),
        ("ROB-001", "Sample Positioning Stage", "motion", "mechanical", "medium", False, None),
    ]
    assets = []
    for tag, name, cat, disc, crit, cal, interval in asset_specs:
        a = Asset(
            tag=tag, name=name, category=cat, discipline=disc, criticality=crit,
            status=random.choice(["operational", "operational", "operational", "degraded", "down"]),
            location_id=random.choice(locations).id,
            manufacturer=random.choice(["Coherent", "Tektronix", "Keysight", "Edwards", "Spectra-Physics"]),
            model=f"M-{random.randint(100, 999)}",
            serial_number=f"SN{random.randint(10000, 99999)}",
            install_date=(now - timedelta(days=random.randint(200, 2000))).date(),
            purchase_cost=random.choice([25000, 80000, 150000, 320000]),
            requires_calibration=cal,
            calibration_interval_days=interval,
            last_calibrated=(now - timedelta(days=random.randint(30, 400))).date() if cal else None,
        )
        db.session.add(a)
        assets.append(a)
    db.session.commit()

    # --- Parts ------------------------------------------------------------
    parts = [
        Part(part_number="FLT-AIR-12", name="HEPA Filter 12in", category="consumable",
             quantity=2, reorder_point=4, unit_cost=85, location="Cabinet 3", vendor="FilterCo"),
        Part(part_number="OIL-VAC-1L", name="Vacuum Pump Oil 1L", category="fluid",
             quantity=6, reorder_point=3, unit_cost=42, location="Cabinet 1", vendor="Edwards"),
        Part(part_number="FUSE-HV-5", name="HV Fuse 5A", category="electrical",
             quantity=1, reorder_point=10, unit_cost=18, location="Drawer A", vendor="DigiKey"),
        Part(part_number="MIRR-25", name="25mm Dielectric Mirror", category="optic",
             quantity=8, reorder_point=2, unit_cost=240, location="Optics Cab", vendor="Thorlabs"),
        Part(part_number="ORING-KIT", name="O-Ring Assortment", category="mechanical",
             quantity=3, reorder_point=2, unit_cost=60, location="Shop", vendor="McMaster"),
    ]
    db.session.add_all(parts)

    # --- Contacts ---------------------------------------------------------
    for u in users + [admin]:
        db.session.add(Contact(
            name=u.full_name, email=u.email, role=u.title,
            discipline=u.discipline, active=True,
        ))
    db.session.commit()

    # --- PM schedules -----------------------------------------------------
    pm_specs = [
        ("Monthly laser optics cleaning", assets[0], 30, 2.0),
        ("Quarterly oscilloscope self-cal", assets[1], 90, 1.0),
        ("Vacuum pump oil change", assets[3], 120, 1.5),
        ("Chiller coolant check", assets[5], 60, 0.5),
        ("Digitizer firmware/health check", assets[7], 180, 1.0),
    ]
    for title, asset, freq, hours in pm_specs:
        next_due = (now + timedelta(days=random.randint(-10, 40))).date()
        db.session.add(PMSchedule(
            title=title, asset_id=asset.id, frequency_days=freq,
            estimated_hours=hours, assignee_id=random.choice(engineers).id,
            next_due=next_due, last_generated=(now - timedelta(days=freq)).date(),
            priority=random.choice(["medium", "high"]),
        ))
    db.session.commit()

    # --- Work orders (historical + open) ---------------------------------
    titles = [
        "Replace failed cooling fan", "Recalibrate detector gain",
        "Investigate beam pointing drift", "Vacuum leak troubleshooting",
        "HV supply intermittent trip", "Align spectrometer slit",
        "Firmware update digitizer", "Replace worn O-rings",
        "Clean optical table surface", "Stage homing fault",
    ]
    failure_codes = ["electrical_fault", "mechanical_wear", "contamination", "software", "alignment"]
    for i in range(45):
        created = now - timedelta(days=random.randint(0, 170), hours=random.randint(0, 23))
        wo_type = random.choices(WO_TYPE_POOL, weights=[5, 3, 2, 1, 1])[0]
        asset = random.choice(assets)
        completed = None
        started = None
        status = random.choices(
            ["completed", "completed", "open", "in_progress", "on_hold"],
            weights=[5, 4, 2, 2, 1],
        )[0]
        downtime = 0.0
        labor = 0.0
        cost_parts = 0.0
        cost_labor = 0.0
        fcode = None
        if status == "completed":
            started = created + timedelta(hours=random.randint(1, 12))
            completed = created + timedelta(hours=random.randint(3, 96))
            labor = round(random.uniform(0.5, 12), 1)
            downtime = round(random.uniform(0, 24), 1) if wo_type == "corrective" else 0.0
            cost_parts = round(random.uniform(0, 1500), 2)
            cost_labor = round(labor * 95, 2)
            fcode = random.choice(failure_codes) if wo_type == "corrective" else None
        due = created + timedelta(days=random.randint(2, 21))
        wo = WorkOrder(
            number=f"WO-{1000 + i}",
            title=random.choice(titles),
            description="Auto-generated demo work order.",
            wo_type=wo_type,
            priority=random.choice(["critical", "high", "medium", "medium", "low"]),
            status=status,
            asset_id=asset.id,
            assignee_id=random.choice(engineers).id,
            requested_by=random.choice(["jchen", "mrivera", "lab.ops@lab.doe.local"]),
            created_at=created, due_date=due, started_at=started, completed_at=completed,
            labor_hours=labor, downtime_hours=downtime,
            cost_parts=cost_parts, cost_labor=cost_labor, failure_code=fcode,
            resolution="Resolved and verified." if completed else None,
        )
        db.session.add(wo)
    db.session.commit()

    # --- Projects + updates ----------------------------------------------
    project_specs = [
        ("DX-101", "Streak Camera Timing Upgrade", "upgrade", "optical", "active", "on_track", 65),
        ("DX-102", "Neutron Diagnostic Characterization", "characterization", "electrical", "active", "at_risk", 40),
        ("DX-103", "Compact Spectrometer R&D", "rnd", "optical", "active", "on_track", 25),
        ("DX-104", "Pulsed Power Reliability Study", "study", "electrical", "on_hold", "off_track", 15),
        ("DX-105", "Automated Alignment System", "build", "mechanical", "planning", "on_track", 5),
    ]
    for code, name, ptype, disc, status, health, pct in project_specs:
        p = Project(
            code=code, name=name, project_type=ptype, discipline=disc,
            status=status, health=health, percent_complete=pct,
            lead_id=random.choice(engineers).id,
            budget=random.choice([50000, 120000, 250000]),
            spent=random.randint(5000, 100000),
            start_date=(now - timedelta(days=random.randint(60, 300))).date(),
            target_date=(now + timedelta(days=random.randint(-30, 200))).date(),
            description="Advanced diagnostics engineering effort.",
        )
        db.session.add(p)
        db.session.flush()
        for j in range(random.randint(2, 4)):
            db.session.add(ProjectUpdate(
                project_id=p.id, author_id=random.choice(engineers).id,
                update_type=random.choice(["note", "milestone", "metric", "risk"]),
                title=f"Update {j + 1}",
                body="Progress recorded during weekly review.",
                metric_name=random.choice(["signal_to_noise", "timing_jitter_ps", "throughput"]),
                metric_value=round(random.uniform(1, 100), 2),
                metric_unit=random.choice(["dB", "ps", "%"]),
                percent_complete=pct,
                created_at=now - timedelta(days=random.randint(1, 60)),
            ))
    db.session.commit()

    # --- Design-project progress board -----------------------------------
    # (representative rows in the style of the FY design-progress sheet)
    design_rows = [
        # number, title, design_name, windchill, epdm, program, DA, pm, dm, dtl,
        #   da_po, status, req, ip, cdr, dr, dwg, tpr, audit, accepted, by, doc_loc
        (1, "OPSPEC", "OPSPEC Work Continuation", "L-CA-5008-01", "", "SEO/LLNL",
         "Wallace", "Wallace", "Wallace", "", "D/A", "A", 1, 0, 0, 0, 3, 5, 5, "N", "", "S"),
        (2, "Target Assembly - LAZE", "Mechanical Target Assembly", "L-RS-5043-01", "",
         "SEO/S&T", "Dutra", "Mazotti", "Dzenitis", "", "D/A", "A", 0, 0, 0, 0, 3, 0, 5, "N", "", "S"),
        (3, "Mid-IR Rack", "Mid-IR Work Continuation", "L-EO-5046-01", "", "SEO",
         "Dutra", "Mazotti", "Dzenitis", "", "D/A", "A", 3, 3, 5, 1, 3, 0, 5, "N", "", "S"),
        (4, "SPLe", "Enhanced Short Pulse Laser Laboratory", "L-PE-635-01", "", "HEDE/S&T",
         "Larsen", "Larsen", "Fornes", "", "D/A P/O", "A", 2, 2, 0, 0, 0, 0, 5, "Y", "JAM", "eP"),
        (6, "BAMS Upgrade", "BAMS Characterization System Upgrade", "", "", "SEO",
         "McCumber", "McCumber", "McCumber", "", "D/A P/O", "A", 3, 3, 3, 3, 3, 3, 3, "Y", "JAM", "WC"),
        (11, "BEEFI", "Big Explosives Experimentation Facility Imaging", "H-OE-1110-01", "",
         "LAO", "Krubsack", "Zepeda", "Kauffman", "", "D/A", "A", 0, 0, 1, 0, 0, 0, 5, "N", "", "eP"),
        (12, "Chilled Door Selection", "H-PP-776-01_08.3", "H-PP-776-01", "", "LANL",
         "Tuzel", "Unk", "Tuzel", "", "D/A", "A", 5, 5, 5, 5, 2, 5, 5, "Y", "JAM", "eP"),
        (14, "CYGNUS", "Engineering Support", "H-PP-17", "", "SEO",
         "Fiscus", "John Smith", "Flores", "", "D/A", "A", 1, 0, 0, 1, 1, 0, 5, "Y", "JAM", "eP"),
        (30, "Excalibur", "Excalibur optical assembly and probes", "H-OE-1117-01", "", "SEO",
         "Esquibel", "Leak", "Smith", "", "D/A", "A", 1, 0, 5, 5, 2, 5, 5, "Y", "JAM", "eP"),
        (31, "Kraken", "Kraken V1 and Kraken V2", "H-CA-750-01", "", "SEO",
         "Lewis", "Pegram", "Smith", "", "D/A", "A", 1, 0, 5, 5, 2, 5, 2, "Y", "JAM", "eP"),
        (34, "BEEF MGDS", "BEEF AEC Methane Gas Delivery System", "L-BF-5057-01", "", "SEO",
         "Bishop", "McCumber", "McCumber", "", "D/A P/O", "I", 3, 3, 3, 3, 3, 3, 3, "Y", "JAM", "WC"),
        (35, "FCBS Bulb Box", "Bulb Box Upgrade", "L-PG-718-01", "", "LLNL",
         "Larsen", "Guyton", "", "", "D/A", "A", 0, 0, 1, 0, 1, 0, 5, "N", "", "WC"),
    ]
    for row in design_rows:
        (num, title, dname, wc, ep, prog, da, pm_, dm_, dtl_, dapo, st,
         req, ip, cdr, dr, dwg, tpr, audit, acc, accby, dl) = row
        db.session.add(DesignProject(
            number=num, title=title, design_name=dname, windchill_number=wc or None,
            epdm_number=ep or None, program=prog, design_authority=da, pm=pm_,
            dm=dm_ or None, dtl=dtl_ or None, da_po=dapo, status=st,
            req=req, ip=ip, cdr=cdr, dr=dr, dwg=dwg, tpr=tpr, audit=audit,
            accepted=acc, accepted_by=accby or None, doc_location=dl,
        ))
    db.session.commit()

    print("Demo data seeded successfully.")


WO_TYPE_POOL = ["corrective", "preventive", "calibration", "inspection", "project"]


if __name__ == "__main__":
    from . import create_app

    app = create_app()
    with app.app_context():
        seed_demo()
