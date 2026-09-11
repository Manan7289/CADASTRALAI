"""Local dev server for CadastraAI's static frontend.

The frontend now fetches everything (buildings, layers, metrics, reports)
from plain files under frontend/data/ -- the same files Vercel serves
statically in production (see build_static.py). This app just mirrors that
locally with Flask instead of `python -m http.server`, so `python app.py`
and the deployed site behave identically.

Run order: fetch_data.py -> fetch_imagery.py -> model.py -> build_static.py -> app.py
"""
from pathlib import Path

from flask import Flask, send_from_directory

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = Flask(__name__, static_folder=None)


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/vendor/<path:filename>")
def vendor(filename):
    return send_from_directory(FRONTEND_DIR / "vendor", filename)


@app.route("/data/<path:filename>")
def data_files(filename):
    return send_from_directory(FRONTEND_DIR / "data", filename)


if __name__ == "__main__":
    app.run(debug=True, port=5050)
