from functools import wraps

from flask import Blueprint, render_template, redirect, url_for, flash
from flask_login import login_required, current_user

from .database import db, User, PredictionHistory, ContactMessage

admin = Blueprint("admin", __name__, url_prefix="/admin")


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for("auth.login"))

        if not current_user.is_admin:
            flash("Access denied. Admins only.", "danger")
            return redirect(url_for("main.dashboard"))

        return f(*args, **kwargs)
    return decorated_function


@admin.route("/")
@login_required
@admin_required
def admin_dashboard():
    users = User.query.order_by(User.created_at.desc()).all()
    predictions = PredictionHistory.query.order_by(PredictionHistory.created_at.desc()).all()
    messages = ContactMessage.query.order_by(ContactMessage.created_at.desc()).all()

    total_users = User.query.count()
    total_admins = User.query.filter_by(is_admin=True).count()
    total_predictions = PredictionHistory.query.count()

    depressed_count = PredictionHistory.query.filter(
        PredictionHistory.prediction.ilike("%depressed%")
    ).count()

    not_depressed_count = PredictionHistory.query.filter(
        PredictionHistory.prediction.ilike("%not depressed%")
    ).count()

    return render_template(
        "admin_dashboard.html",
        users=users,
        predictions=predictions,
        messages=messages,
        total_users=total_users,
        total_admins=total_admins,
        total_predictions=total_predictions,
        depressed_count=depressed_count,
        not_depressed_count=not_depressed_count
    )


@admin.route("/delete_user/<int:user_id>", methods=["POST"])
@login_required
@admin_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)

    if user.id == current_user.id:
        flash("You cannot delete your own admin account.", "warning")
        return redirect(url_for("admin.admin_dashboard"))

    PredictionHistory.query.filter_by(user_id=user.id).delete()
    db.session.delete(user)
    db.session.commit()

    flash("User deleted successfully.", "success")
    return redirect(url_for("admin.admin_dashboard"))


@admin.route("/delete_prediction/<int:prediction_id>", methods=["POST"])
@login_required
@admin_required
def delete_prediction(prediction_id):
    prediction = PredictionHistory.query.get_or_404(prediction_id)
    db.session.delete(prediction)
    db.session.commit()

    flash("Prediction record deleted successfully.", "success")
    return redirect(url_for("admin.admin_dashboard"))