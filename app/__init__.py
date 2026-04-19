import os
from pathlib import Path

from flask import Flask, redirect, url_for
from flask_login import current_user

from .auth import auth_bp, bcrypt, csrf, login_manager
from .blueprints.admin import admin_bp
from .models import User, db


def _seed_default_superadmin() -> None:
    """Create default superadmin when database has no users."""
    if User.query.count() > 0:
        return

    default_admin = User(
        name="Admin",
        email="admin@campus.local",
        password_hash=bcrypt.generate_password_hash("changeme123").decode("utf-8"),
        role="superadmin",
        rfid_uid=None,
        is_active=1,
    )
    db.session.add(default_admin)
    db.session.commit()


def create_app(test_config: dict | None = None) -> Flask:
    project_root = Path(__file__).resolve().parent.parent
    default_db_uri = f"sqlite:///{project_root / 'lpr_system.db'}"

    app = Flask(
        __name__,
        template_folder=str(project_root / "templates"),
    )

    app.config.from_mapping(
        SECRET_KEY=os.getenv("SECRET_KEY", "dev-change-this-secret"),
        SQLALCHEMY_DATABASE_URI=os.getenv("DATABASE_URL", default_db_uri),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )

    if test_config:
        app.config.update(test_config)

    db.init_app(app)
    bcrypt.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)

    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp)

    @app.route("/")
    def home():
        if not current_user.is_authenticated:
            return redirect(url_for("auth.login"))
        if current_user.role in {"superadmin", "guard"}:
            return redirect(url_for("admin.users_index"))
        return redirect(url_for("auth.profile"))

    @app.errorhandler(403)
    def forbidden(_error):
        if current_user.is_authenticated:
            return redirect(url_for("auth.profile"))
        return redirect(url_for("auth.login"))

    with app.app_context():
        db.create_all()
        _seed_default_superadmin()

    return app
