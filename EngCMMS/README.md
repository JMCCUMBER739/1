# EngCMMS — Combined CMMS + Engineering Data Tracker

A self-hosted, **100% Python** maintenance-management and engineering-data
platform for laboratory teams running R&D and characterization on advanced
diagnostics. It runs entirely on your own machine or internal server — **no
external/cloud CMMS, no internet dependency** — which makes it suitable for
DOE / closed-network environments.

It combines the core capabilities of leading commercial CMMS products
(Fiix, Limble, UpKeep, MaintainX, IBM Maximo) with an engineering project /
data dashboard, plus auto-generated email notifications and a forms/procedure
repository.

---

## Why this design

Commercial CMMS tools converge on the same proven feature set. EngCMMS
re-implements and improves on that core:

| Commercial CMMS feature | EngCMMS implementation + improvement |
|---|---|
| Asset/equipment registry | Assets with hierarchy, criticality, location **+ calibration tracking** (critical for diagnostics labs) |
| Work orders | Corrective / preventive / calibration / inspection WOs with cost, labor, downtime & failure codes |
| Preventive maintenance | PM schedules with one-click "generate due work orders" |
| Maintenance requests | Manual **or email-sourced** intake → one-click convert to work order |
| Inventory / spare parts | Stock levels, reorder points, stock value, quick adjust |
| Reporting / KPIs | MTTR, MTBF, PM compliance, availability, cost & downtime analytics |
| Document management | Procedures / forms / manuals / drawings repository with attachments |
| Notifications | Auto-generated emails + editable templates + subscription-based routing |
| — *(added)* | **Engineering project & data tracker** with metric time-series, health/KPIs |

---

## Feature overview

- **Intuitive dashboard** — KPI cards (open WOs, availability, PM compliance,
  MTTR, MTBF, calibrations due, projects at risk, low stock) plus interactive
  charts (work-order flow, status/priority/type breakdowns, asset status,
  project health) and an "attention" panel.
- **Advanced analytics** — 12-month created-vs-completed trend, maintenance
  cost trend (parts vs labor), top-downtime-by-asset, open workload by
  assignee, and more.
- **Work orders** — full lifecycle, metrics, attachments, quick status, and
  **auto-email to the assignee + subscribers** on assignment.
- **Preventive maintenance** — recurring schedules; generate work orders for
  everything that is due in one click.
- **Assets + calibration** — equipment registry with calibration intervals and
  overdue tracking.
- **Inventory** — spare parts with reorder alerts and stock valuation.
- **Requests** — capture issues manually or paste in email requests; triage and
  convert to work orders.
- **Document repository** — upload and attach procedures, forms, manuals,
  drawings; link them to assets/work orders/projects.
- **Engineering projects** — track R&D / characterization / build / study work
  with progress, budget, health, **logged metrics that render as time-series
  charts**, and an activity log.
- **Email** — auto-generate addresses from names (`first.last@domain`), manage
  notification **contacts** with per-category subscriptions, edit reusable
  **templates** with `{{ placeholder }}` tokens, compose ad-hoc messages, and
  send a **one-click weekly digest** populated from live KPIs.
- **Roles & permissions** — `admin` (full control + manual data entry),
  `engineer` / `technician` (create & edit), `viewer` (read-only).
- **Email "preview mode"** — with no SMTP configured, every generated email is
  fully rendered and logged but **not sent** — safe for local office testing.

---

## Where data is stored (`D:\EngCMMS`)

The application **code** is separate from its **data**. The SQLite database and
all uploaded attachments are written to a data directory, resolved as:

1. `ENGCMMS_DATA_DIR` environment variable, if set.
2. **`D:\EngCMMS`** automatically, on Windows.
3. `~/EngCMMS`, on Linux/macOS.

So on your Windows office machine, just run it and everything lands in
`D:\EngCMMS\engcmms.db` and `D:\EngCMMS\uploads\` — no configuration needed.

---

## Quick start (local office testing — Windows)

```bat
:: 1. Get the code into a folder, e.g. C:\Apps\EngCMMS, then:
cd C:\Apps\EngCMMS

:: 2. Create a virtual environment and install dependencies
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

:: 3. (optional) Load sample data so the dashboard isn't empty
python run.py --seed

:: ...or just start it (creates an empty D:\EngCMMS database):
python run.py
```

Then open <http://127.0.0.1:5000> and sign in with **`admin` / `admin`**
(change this immediately in **Admin → Users**).

### Quick start (Linux/macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py --seed       # http://127.0.0.1:5000
```

---

## Deploying on an internal server

Use a production WSGI server (do **not** use `python run.py` in production).

**Linux:**
```bash
pip install -r requirements.txt
ENGCMMS_DATA_DIR=/srv/engcmms-data \
ENGCMMS_ENV=production \
gunicorn -w 4 -b 0.0.0.0:8000 wsgi:app
```

**Windows (e.g. behind IIS or as a service):**
```bat
pip install -r requirements.txt
waitress-serve --listen=0.0.0.0:8000 wsgi:app
```

Set environment variables (or copy `.env.example` to `.env`) to point at shared
storage, enable your SMTP relay, set a fixed `ENGCMMS_SECRET_KEY`, and brand the
app. See `.env.example` for every option.

### Enabling real email

Set `ENGCMMS_MAIL_SERVER` (your internal relay) and, if needed,
`ENGCMMS_MAIL_PORT`, `ENGCMMS_MAIL_USE_TLS`, `ENGCMMS_MAIL_USERNAME`,
`ENGCMMS_MAIL_PASSWORD`, and `ENGCMMS_MAIL_SENDER`. Until configured, the system
stays in safe preview mode.

---

## Configuration reference

| Variable | Purpose | Default |
|---|---|---|
| `ENGCMMS_DATA_DIR` | DB + uploads location | `D:\EngCMMS` (Win) / `~/EngCMMS` |
| `ENGCMMS_SECRET_KEY` | Flask session key | auto-generated, stored in data dir |
| `ENGCMMS_ADMIN_PASSWORD` | Initial admin password | `admin` |
| `ENGCMMS_ORG_NAME` | Branding | Advanced Diagnostics Engineering |
| `ENGCMMS_EMAIL_DOMAIN` | Domain for auto-generated addresses | `lab.doe.local` |
| `ENGCMMS_MAIL_SERVER` … | SMTP settings | unset → preview mode |
| `ENGCMMS_DATABASE_URI` | Override DB (e.g. PostgreSQL) | SQLite in data dir |
| `ENGCMMS_MAX_UPLOAD_MB` | Max attachment size | 32 |
| `ENGCMMS_ENV` | `development` / `production` | production |

---

## Project layout

```
EngCMMS/
├── run.py                 # local dev launcher (python run.py [--seed])
├── wsgi.py                # production entry point (gunicorn/waitress)
├── requirements.txt
├── .env.example
├── sample_forms/          # example form to upload into the repository
└── engcmms/
    ├── __init__.py        # application factory
    ├── config.py          # config + data-dir resolution
    ├── extensions.py
    ├── models.py          # all database models
    ├── permissions.py     # role decorators
    ├── analytics.py       # KPI / analytics engine
    ├── email_service.py   # templating, auto-generation, SMTP/preview
    ├── seed.py            # bootstrap defaults + demo data
    ├── blueprints/        # routes per module
    ├── templates/         # Jinja2 UI
    └── static/            # CSS, JS, vendored Chart.js (offline)
```

---

## Default roles

| Role | Capabilities |
|---|---|
| **admin** | Everything, incl. user management & manual data entry |
| **engineer** | Create/edit work orders, assets, PM, projects, docs |
| **technician** | Create/edit work orders, assets, PM, projects, docs |
| **viewer** | Read-only access to dashboards and records |

Demo accounts created by `--seed` use the password `password`
(e.g. `jchen`, `mrivera`, `tsato`, `kpatel`, `lwagner`, `viewer`).

---

## Notes

- No internet access is required at runtime; Chart.js is vendored locally.
- SQLite is the default and is perfect for local + small-team server use. For a
  larger deployment, point `ENGCMMS_DATABASE_URI` at PostgreSQL.
- Back up the data directory (`D:\EngCMMS`) to back up everything.
