from transformers import pipeline

# Load model once
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)


def predict_text(text):
    result = classifier(text)[0]

    sentiment = result["label"]   # POSITIVE / NEGATIVE
    score = result["score"]

    if sentiment == "POSITIVE":
        positive = score
        negative = 1 - score
    else:
        negative = score
        positive = 1 - score

    risk_score = negative

    if risk_score > 0.50:
        prediction = "Depressed"
    else:
        prediction = "Not Depressed"

    if risk_score > 0.75:
        risk_label = "High Risk"
    elif risk_score > 0.50:
        risk_label = "Moderate Risk"
    else:
        risk_label = "Low Risk"

    return prediction, sentiment, positive, negative, risk_score, risk_label