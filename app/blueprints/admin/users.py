import math
import re

from flask import flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required
from sqlalchemy import func, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from ...auth import role_required, bcrypt
from ...models import db, User, Vehicle
from . import admin_bp


ROLE_CHOICES = ("superadmin", "guard", "user")
PLATE_REGEX = re.compile(r"^[A-Z]{2,3}\s?\d{3,4}$")


def _normalize_email(value: str) -> str:
    return (value or "").strip().lower()


def _normalize_plate(value: str) -> str:
    compact = re.sub(r"\s+", "", (value or "").strip().upper())
    return compact


def _allowed_roles_for_creator(creator_role: str) -> tuple[str, ...]:
    role = str(creator_role or "").lower()
    if role == "guard":
        return ("user",)
    return ROLE_CHOICES


def _extract_plate_numbers(form_data) -> tuple[list[str], list[str]]:
    """Validate and normalize inline vehicle plate list from the form."""
    raw_values = form_data.getlist("plate_numbers")
    errors: list[str] = []
    plates: list[str] = []
    seen: set[str] = set()

    for raw in raw_values:
        candidate = (raw or "").strip().upper()
        if not candidate:
            continue

        candidate = re.sub(r"\s+", " ", candidate)
        if not PLATE_REGEX.fullmatch(candidate):
            errors.append(
                f"Invalid plate format: {candidate}. "
                "Expected PH format like AB1234 or ABC 1234."
            )
            continue

        normalized = _normalize_plate(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        plates.append(normalized)

    if not plates:
        errors.append("At least one valid vehicle plate number is required.")

    return plates, errors


def _email_in_use(email: str, exclude_user_id: int | None = None) -> bool:
    query = User.query.filter(func.lower(User.email) == email.lower())
    if exclude_user_id is not None:
        query = query.filter(User.id != exclude_user_id)
    return query.first() is not None


def _render_form(
    mode: str,
    user: User | None,
    form_data: dict,
    vehicles: list[str],
    roles: tuple[str, ...] = ROLE_CHOICES,
):
    return render_template(
        "admin/users/form.html",
        mode=mode,
        user=user,
        form_data=form_data,
        vehicles=vehicles if vehicles else [""],
        roles=roles,
    )


@admin_bp.route("/users", methods=["GET"])
@login_required
@role_required("superadmin", "guard")
def users_index():
    q = request.args.get("q", "").strip()
    role_filter = request.args.get("role", "").strip().lower()

    try:
        page = max(int(request.args.get("page", "1")), 1)
    except ValueError:
        page = 1

    per_page = 10

    query = User.query.options(selectinload(User.vehicles))

    if q:
        like = f"%{q}%"
        query = query.filter(
            or_(
                User.name.ilike(like),
                User.email.ilike(like),
                User.role.ilike(like),
            )
        )

    if role_filter in ROLE_CHOICES:
        query = query.filter(User.role == role_filter)

    total = query.count()
    pages = max(1, math.ceil(total / per_page)) if total else 1
    if page > pages:
        page = pages

    users = (
        query.order_by(User.created_at.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )

    return render_template(
        "admin/users/index.html",
        users=users,
        q=q,
        role_filter=role_filter,
        page=page,
        per_page=per_page,
        pages=pages,
        total=total,
        can_create=(current_user.role in {"superadmin", "guard"}),
        can_manage=(current_user.role == "superadmin"),
    )


@admin_bp.route("/users/new", methods=["GET"])
@login_required
@role_required("superadmin", "guard")
def users_new():
    allowed_roles = _allowed_roles_for_creator(current_user.role)
    form_data = {
        "name": "",
        "email": "",
        "role": allowed_roles[0],
        "rfid_uid": "",
        "is_active": True,
    }
    return _render_form(
        mode="create",
        user=None,
        form_data=form_data,
        vehicles=[""],
        roles=allowed_roles,
    )


@admin_bp.route("/users/new", methods=["POST"])
@login_required
@role_required("superadmin", "guard")
def users_create():
    allowed_roles = _allowed_roles_for_creator(current_user.role)
    name = request.form.get("name", "").strip()
    email = _normalize_email(request.form.get("email", ""))
    password = request.form.get("password", "")
    role = request.form.get("role", "").strip().lower()
    rfid_uid = request.form.get("rfid_uid", "").strip().upper() or None
    is_active = 1 if request.form.get("is_active") == "on" else 0

    plates, plate_errors = _extract_plate_numbers(request.form)

    errors: list[str] = []
    if not name:
        errors.append("Name is required.")
    if not email:
        errors.append("Email is required.")
    if not password:
        errors.append("Password is required.")
    if role not in allowed_roles:
        errors.append("Invalid role selected.")
    if email and _email_in_use(email):
        errors.append("That email is already registered.")
    errors.extend(plate_errors)

    form_data = {
        "name": name,
        "email": email,
        "role": role if role in allowed_roles else allowed_roles[0],
        "rfid_uid": rfid_uid or "",
        "is_active": bool(is_active),
    }

    if errors:
        for err in errors:
            flash(err, "error")
        return _render_form(
            mode="create",
            user=None,
            form_data=form_data,
            vehicles=plates,
            roles=allowed_roles,
        )

    password_hash = bcrypt.generate_password_hash(password).decode("utf-8")

    user = User(
        name=name,
        email=email,
        password_hash=password_hash,
        role=role,
        rfid_uid=rfid_uid,
        is_active=is_active,
    )

    for plate in plates:
        user.vehicles.append(Vehicle(plate_number=plate))

    db.session.add(user)

    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        flash("Unable to create user due to a database constraint.", "error")
        return _render_form(
            mode="create",
            user=None,
            form_data=form_data,
            vehicles=plates,
            roles=allowed_roles,
        )

    flash("User created successfully.", "success")
    return redirect(url_for("admin.users_index"))


@admin_bp.route("/users/<int:user_id>/edit", methods=["GET"])
@login_required
@role_required("superadmin")
def users_edit(user_id: int):
    user = User.query.options(selectinload(User.vehicles)).filter_by(id=user_id).first_or_404()

    form_data = {
        "name": user.name,
        "email": user.email,
        "role": user.role,
        "rfid_uid": user.rfid_uid or "",
        "is_active": bool(user.is_active),
    }
    vehicles = [v.plate_number for v in user.vehicles] or [""]

    return _render_form(mode="edit", user=user, form_data=form_data, vehicles=vehicles)


@admin_bp.route("/users/<int:user_id>/edit", methods=["POST"])
@login_required
@role_required("superadmin")
def users_update(user_id: int):
    user = User.query.options(selectinload(User.vehicles)).filter_by(id=user_id).first_or_404()

    name = request.form.get("name", "").strip()
    email = _normalize_email(request.form.get("email", ""))
    password = request.form.get("password", "")
    role = request.form.get("role", "").strip().lower()
    rfid_uid = request.form.get("rfid_uid", "").strip().upper() or None
    is_active = 1 if request.form.get("is_active") == "on" else 0

    plates, plate_errors = _extract_plate_numbers(request.form)

    errors: list[str] = []
    if not name:
        errors.append("Name is required.")
    if not email:
        errors.append("Email is required.")
    if role not in ROLE_CHOICES:
        errors.append("Invalid role selected.")
    if email and _email_in_use(email, exclude_user_id=user.id):
        errors.append("That email is already registered.")
    errors.extend(plate_errors)

    form_data = {
        "name": name,
        "email": email,
        "role": role if role in ROLE_CHOICES else user.role,
        "rfid_uid": rfid_uid or "",
        "is_active": bool(is_active),
    }

    if errors:
        for err in errors:
            flash(err, "error")
        return _render_form(mode="edit", user=user, form_data=form_data, vehicles=plates)

    user.name = name
    user.email = email
    user.role = role
    user.rfid_uid = rfid_uid
    user.is_active = is_active

    if password:
        user.password_hash = bcrypt.generate_password_hash(password).decode("utf-8")

    # Replace vehicle list with current form values.
    user.vehicles.clear()
    for plate in plates:
        user.vehicles.append(Vehicle(plate_number=plate))

    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        flash("Unable to update user due to a database constraint.", "error")
        return _render_form(mode="edit", user=user, form_data=form_data, vehicles=plates)

    flash("User updated successfully.", "success")
    return redirect(url_for("admin.users_index"))


@admin_bp.route("/users/<int:user_id>/delete", methods=["POST"])
@login_required
@role_required("superadmin")
def users_delete(user_id: int):
    user = User.query.filter_by(id=user_id).first_or_404()

    if user.id == current_user.id:
        flash("You cannot deactivate your own account.", "error")
        return redirect(url_for("admin.users_index"))

    user.is_active = 0
    db.session.commit()

    flash("User deactivated successfully.", "success")
    return redirect(url_for("admin.users_index"))
