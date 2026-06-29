"""Local launcher for EngCMMS.

Usage:
    python run.py            # start the development server on http://127.0.0.1:5000
    python run.py --seed     # populate realistic demo data, then start
"""

from __future__ import annotations

import argparse

from engcmms import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="EngCMMS server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--seed", action="store_true", help="seed demo data before launch")
    args = parser.parse_args()

    app = create_app()

    if args.seed:
        from engcmms.seed import seed_demo

        with app.app_context():
            seed_demo()

    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
