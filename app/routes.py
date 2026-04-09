from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user

from .database import db, PredictionHistory, ContactMessage
from .predictor import predict_text
from .attention import generate_attention
from .visualization import (
    generate_confusion_matrix,
    generate_roc_curve,
    accuracy_graph,
    loss_graph,
    model_comparison_graph
)

main = Blueprint("main", __name__)

# Generate charts once
generate_confusion_matrix()
generate_roc_curve()
accuracy_graph()
loss_graph()
model_comparison_graph()


@main.route("/", methods=["GET"])
def landing():
    return render_template("home.html")


@main.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")


@main.route("/about")
def about():
    return render_template("about.html")


@main.route("/evaluation")
@login_required
def evaluation():
    if not current_user.is_admin:
        flash("Access denied. Admin only.", "danger")
        return redirect(url_for("main.dashboard"))

    return render_template("evaluation.html")


@main.route("/contact", methods=["GET", "POST"])
@login_required
def contact():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        subject = request.form.get("subject", "").strip()
        message = request.form.get("message", "").strip()

        if not name or not email or not subject or not message:
            flash("Please fill all fields.", "danger")
            return redirect(url_for("main.contact"))

        new_message = ContactMessage(
            name=name,
            email=email,
            subject=subject,
            message=message
        )
        db.session.add(new_message)
        db.session.commit()

        flash("Your message has been submitted successfully.", "success")
        return redirect(url_for("main.contact"))

    return render_template("contact.html")


@main.route("/analyze", methods=["GET", "POST"])
@login_required
def analyze():
    if request.method == "GET":
        return render_template("analyze.html")

    text = request.form.get("text", "").strip()

    if not text:
        flash("Please enter some text to analyze.", "danger")
        return redirect(url_for("main.analyze"))

    try:
        prediction, sentiment, positive, negative, risk, risk_label = predict_text(text)
        attention = generate_attention(text)

        positive = round(positive * 100, 2)
        negative = round(negative * 100, 2)
        risk = round(risk * 100, 2)

        # confidence based on final prediction
        if prediction == "Depressed":
            confidence = risk
        else:
            confidence = round(positive, 2)

        history_item = PredictionHistory(
            user_id=current_user.id,
            input_text=text,
            prediction=prediction,
            confidence=confidence,
            risk_score=risk,
            risk_label=risk_label
        )
        db.session.add(history_item)
        db.session.commit()

        return render_template(
            "result.html",
            text=text,
            prediction=prediction,
            sentiment=sentiment,
            positive=positive,
            negative=negative,
            risk=risk,
            risk_label=risk_label,
            confidence=confidence,
            words=text.split(),
            attention=attention
        )

    except Exception as e:
        print("ERROR:", e)
        flash("Something went wrong during prediction.", "danger")
        return redirect(url_for("main.analyze"))


@main.route("/history")
@login_required
def history():
    predictions = PredictionHistory.query.filter_by(user_id=current_user.id) \
        .order_by(PredictionHistory.created_at.desc()) \
        .all()

    return render_template("history.html", predictions=predictions)