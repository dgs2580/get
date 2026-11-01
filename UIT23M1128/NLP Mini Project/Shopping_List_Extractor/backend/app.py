"""Flask backend entry point.   Run:  python backend/app.py"""
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flask import Flask  # noqa: E402
from flask_cors import CORS  # noqa: E402

from backend.routes import api  # noqa: E402
from backend.utils import fail  # noqa: E402
from nlp.preprocessing import ensure_nltk_resources  # noqa: E402

logging.basicConfig(level=logging.INFO)


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB uploads
    CORS(app)
    app.register_blueprint(api)
    ensure_nltk_resources()

    @app.errorhandler(404)
    def _404(_):
        return fail("Endpoint not found.", 404, "not_found")

    @app.errorhandler(405)
    def _405(_):
        return fail("Method not allowed for this endpoint.", 405, "method_not_allowed")

    @app.errorhandler(413)
    def _413(_):
        return fail("Uploaded file is too large (max 2 MB).", 413, "file_too_large")

    @app.errorhandler(Exception)
    def _500(e):
        logging.exception("Unhandled error: %s", e)
        return fail("Something went wrong while processing your request.", 500, "server_error")

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")),
            debug=os.getenv("FLASK_DEBUG", "0") == "1")
