"""WSGI entry point for production servers (gunicorn / waitress / IIS).

Example (Linux internal server):
    gunicorn -w 4 -b 0.0.0.0:8000 wsgi:app

Example (Windows internal server):
    waitress-serve --listen=0.0.0.0:8000 wsgi:app
"""

from engcmms import create_app

app = create_app()

if __name__ == "__main__":
    app.run()
