"""Local launcher for EngCMMS.

Usage:
    python run.py            # start the server on http://127.0.0.1:5000
    python run.py --seed     # populate realistic demo data, then start
    python run.py --reload   # enable Flask's auto-reloader (terminals only)

Note: the auto-reloader is OFF by default because it raises ``SystemExit``
when launched from inside an IDE console such as Spyder. Use ``--reload`` only
when running from a real terminal and you want live code reloading.
"""

from __future__ import annotations

import argparse
import os
import sys

from engcmms import create_app


def _running_in_ide() -> bool:
    """Best-effort detection of Spyder / IPython / notebook kernels."""
    if any(key in os.environ for key in ("SPYDER_ARGS", "SPY_PYTHONPATH")):
        return True
    return "IPython" in sys.modules or "spyder_kernels" in sys.modules


def main() -> None:
    parser = argparse.ArgumentParser(description="EngCMMS server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--seed", action="store_true", help="seed demo data before launch")
    parser.add_argument("--reload", action="store_true",
                        help="enable the auto-reloader (terminals only, not IDE consoles)")
    # parse_known_args so extra args injected by an IDE 'runfile' don't crash us.
    args, _ = parser.parse_known_args()

    app = create_app()

    if args.seed:
        from engcmms.seed import seed_demo

        with app.app_context():
            seed_demo()

    use_reloader = args.reload and not _running_in_ide()

    print(f"\nEngCMMS is starting — open http://{args.host}:{args.port} in your browser.")
    print("Sign in with admin / admin (change it under Admin -> Users).\n")

    app.run(host=args.host, port=args.port, debug=True, use_reloader=use_reloader)


if __name__ == "__main__":
    main()
