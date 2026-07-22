"""Application configuration for EngCMMS.

The data directory (database + uploaded attachments) is intentionally kept
separate from the application code so the program can be deployed read-only on
an internal server while persisting its data elsewhere.

Resolution order for the data directory:
    1. ``ENGCMMS_DATA_DIR`` environment variable (highest priority)
    2. ``D:\\EngCMMS`` on Windows (the location requested by the operator)
    3. ``~/EngCMMS`` on every other platform (handy for local Linux/macOS testing)
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is optional
    pass


def resolve_data_dir() -> Path:
    """Return the directory used to persist the database and uploads."""
    env_dir = os.environ.get("ENGCMMS_DATA_DIR")
    if env_dir:
        data_dir = Path(env_dir)
    elif os.name == "nt":
        data_dir = Path(r"D:\EngCMMS")
    else:
        data_dir = Path.home() / "EngCMMS"

    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "uploads").mkdir(parents=True, exist_ok=True)
    return data_dir


def _resolve_secret_key(data_dir: Path) -> str:
    """Use a stable secret key so sessions survive restarts on a server."""
    env_key = os.environ.get("ENGCMMS_SECRET_KEY")
    if env_key:
        return env_key

    key_file = data_dir / ".secret_key"
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()

    key = secrets.token_hex(32)
    try:
        key_file.write_text(key, encoding="utf-8")
    except OSError:
        # Fall back to an ephemeral key if the data dir is not writable.
        pass
    return key


class Config:
    """Base configuration shared by every environment."""

    DATA_DIR: Path = resolve_data_dir()
    UPLOAD_DIR: Path = DATA_DIR / "uploads"

    SECRET_KEY = _resolve_secret_key(DATA_DIR)

    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "ENGCMMS_DATABASE_URI",
        f"sqlite:///{(DATA_DIR / 'engcmms.db').as_posix()}",
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Maximum attachment size (default 32 MB).
    MAX_CONTENT_LENGTH = int(os.environ.get("ENGCMMS_MAX_UPLOAD_MB", "32")) * 1024 * 1024

    ALLOWED_UPLOAD_EXTENSIONS = {
        "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "csv", "txt",
        "png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff", "svg",
        "dwg", "dxf", "step", "stp", "igs", "zip",
    }

    # ------------------------------------------------------------------
    # SMTP / email settings.  When credentials are not provided the app runs
    # in "preview" mode: generated emails are rendered and logged but not sent,
    # which is ideal for local office testing.
    # ------------------------------------------------------------------
    MAIL_SERVER = os.environ.get("ENGCMMS_MAIL_SERVER", "")
    MAIL_PORT = int(os.environ.get("ENGCMMS_MAIL_PORT", "25"))
    MAIL_USE_TLS = os.environ.get("ENGCMMS_MAIL_USE_TLS", "false").lower() == "true"
    MAIL_USERNAME = os.environ.get("ENGCMMS_MAIL_USERNAME", "")
    MAIL_PASSWORD = os.environ.get("ENGCMMS_MAIL_PASSWORD", "")
    MAIL_DEFAULT_SENDER = os.environ.get(
        "ENGCMMS_MAIL_SENDER", "engcmms-noreply@lab.local"
    )

    # Domain used to auto-generate email addresses from names.
    EMAIL_DOMAIN = os.environ.get("ENGCMMS_EMAIL_DOMAIN", "lab.doe.local")

    ORG_NAME = os.environ.get("ENGCMMS_ORG_NAME", "Advanced Diagnostics Engineering")

    # Shared secret for the machine-to-machine email-intake API. When a mail
    # gateway / IMAP poller posts submissions it must present this token.
    INTAKE_TOKEN = os.environ.get("ENGCMMS_INTAKE_TOKEN", "")


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


def get_config() -> type[Config]:
    env = os.environ.get("ENGCMMS_ENV", "production").lower()
    return DevelopmentConfig if env in {"dev", "development"} else ProductionConfig
