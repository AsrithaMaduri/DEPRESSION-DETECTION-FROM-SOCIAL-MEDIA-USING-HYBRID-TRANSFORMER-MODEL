import re
from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.exc import IntegrityError

from .database import db, User

auth = Blueprint("auth", __name__)


@auth.route("/register", methods=["GET", "POST"])
def register():
    username = ""
    email = ""

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()

        errors = []

        # Username validation
        if not username:
            errors.append("Username is required.")

        # Email validation
        if not email:
            errors.append("Email is required.")
        else:
            email_pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
            if not re.match(email_pattern, email):
                errors.append("Please enter a valid email address.")

        # Password validation
        if not password:
            errors.append("Password is required.")
        elif len(password) < 6:
            errors.append("Password must be at least 6 characters.")

        # Duplicate username
        existing_username = User.query.filter_by(username=username).first()
        if username and existing_username:
            errors.append("Username is already taken.")

        # Duplicate email
        existing_email = User.query.filter_by(email=email).first()
        if email and existing_email:
            errors.append("Email is already registered.")

        if errors:
            for error in errors:
                flash(error, "danger")
            return render_template(
                "register.html",
                username=username,
                email=email
            )

        try:
            hashed_password = generate_password_hash(password)

            new_user = User(
                username=username,
                email=email,
                password=hashed_password
            )

            db.session.add(new_user)
            db.session.commit()

            flash("Account created successfully. Please login.", "success")
            return redirect(url_for("auth.login"))

        except IntegrityError:
            db.session.rollback()
            flash("Username or email already exists.", "danger")
            return render_template(
                "register.html",
                username=username,
                email=email
            )

    return render_template("register.html", username=username, email=email)


@auth.route("/login", methods=["GET", "POST"])
def login():
    email = ""

    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()

        if not email:
            flash("Email is required.", "danger")
            return render_template("login.html", email=email)

        if not password:
            flash("Password is required.", "danger")
            return render_template("login.html", email=email)

        user = User.query.filter_by(email=email).first()

        if not user:
            flash("No account found with this email.", "danger")
            return render_template("login.html", email=email)

        if not check_password_hash(user.password, password):
            flash("Incorrect password.", "danger")
            return render_template("login.html", email=email)

        login_user(user)
        flash("Login successful.", "success")
        return redirect(url_for("main.dashboard"))

    return render_template("login.html", email=email)


@auth.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.", "success")
    return redirect(url_for("main.landing"))