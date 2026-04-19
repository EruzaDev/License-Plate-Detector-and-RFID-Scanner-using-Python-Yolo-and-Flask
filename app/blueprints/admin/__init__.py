from flask import Blueprint


admin_bp = Blueprint("admin", __name__, url_prefix="/admin")

# Import routes so they are registered on blueprint import.
from . import users  # noqa: E402,F401
