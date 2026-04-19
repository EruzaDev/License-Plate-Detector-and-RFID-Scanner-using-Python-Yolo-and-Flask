from functools import wraps

from flask import Blueprint, flash, redirect, render_template, request, url_for, abort
from flask_bcrypt import Bcrypt
from flask_login import (
    LoginManager,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from flask_wtf.csrf import CSRFProtect
from sqlalchemy import func

from .models import User, db


auth_bp = Blueprint("auth", __name__)

login_manager = LoginManager()
bcrypt = Bcrypt()
csrf = CSRFProtect()

login_manager.login_view = "auth.login"
login_manager.login_message_category = "warning"


@login_manager.user_loader
def load_user(user_id: str):
    try:
        return db.session.get(User, int(user_id))
    except (TypeError, ValueError):
        return None


def role_required(*allowed_roles: str):
    """Ensure the current user has one of the required roles."""
    allowed = {role.lower() for role in allowed_roles}

    def decorator(view_func):
        @wraps(view_func)
        def wrapped(*args, **kwargs):
            if not current_user.is_authenticated:
                return login_manager.unauthorized()
            if not bool(current_user.is_active):
                abort(403)
            if allowed and str(current_user.role).lower() not in allowed:
                abort(403)
            return view_func(*args, **kwargs)

        return wrapped

    return decorator


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        if current_user.role in {"superadmin", "guard"}:
            return redirect(url_for("admin.users_index"))
        logout_user()
        flash("Your account role is not allowed to sign in here.", "error")
        return redirect(url_for("auth.login"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Email and password are required.", "error")
            return render_template("auth/login.html")

        user = User.query.filter(func.lower(User.email) == email).first()
        if user is None or not bcrypt.check_password_hash(user.password_hash, password):
            flash("Invalid email or password.", "error")
            return render_template("auth/login.html")

        if not bool(user.is_active):
            flash("This account is inactive. Contact a superadmin.", "error")
            return render_template("auth/login.html")

        if str(user.role).lower() not in {"superadmin", "guard"}:
            flash("Your account role is not allowed to sign in here.", "error")
            return render_template("auth/login.html")

        login_user(user)
        flash("Login successful.", "success")

        next_url = request.args.get("next", "").strip()
        if next_url.startswith("/") and not next_url.startswith("//"):
            return redirect(next_url)

        return redirect(url_for("admin.users_index"))

    return render_template("auth/login.html")


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("auth.login"))


@auth_bp.route("/profile")
@login_required
def profile():
    return render_template("profile.html")
