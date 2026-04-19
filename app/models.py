from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.Text, nullable=False)
    email = db.Column(db.Text, unique=True, nullable=False, index=True)
    password_hash = db.Column(db.Text, nullable=False)
    role = db.Column(db.Text, nullable=False)  # superadmin | guard | user
    rfid_uid = db.Column(db.Text, nullable=True)
    is_active = db.Column(db.Integer, default=1, nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.current_timestamp())

    vehicles = db.relationship(
        "Vehicle",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    @property
    def active(self) -> bool:
        return bool(self.is_active)

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email} role={self.role}>"


class Vehicle(db.Model):
    __tablename__ = "vehicles"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    plate_number = db.Column(db.Text, nullable=False)
    registered_at = db.Column(db.DateTime, server_default=db.func.current_timestamp())

    user = db.relationship("User", back_populates="vehicles")

    def __repr__(self) -> str:
        return f"<Vehicle id={self.id} plate={self.plate_number}>"
