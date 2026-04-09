import os
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from wordcloud import WordCloud

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(BASE_DIR, "static", "charts")
os.makedirs(CHART_DIR, exist_ok=True)


def accuracy_graph():
    epochs = [1, 2, 3, 4, 5]
    accuracy = [0.70, 0.78, 0.84, 0.89, 0.92]

    plt.figure()
    plt.plot(epochs, accuracy, marker="o", label="Accuracy")
    plt.title("Accuracy Graph")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    path = os.path.join(CHART_DIR, "accuracy_graph.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def loss_graph():
    epochs = [1, 2, 3, 4, 5]
    loss = [0.65, 0.52, 0.40, 0.30, 0.21]

    plt.figure()
    plt.plot(epochs, loss, marker="o", label="Loss")
    plt.title("Loss Graph")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    path = os.path.join(CHART_DIR, "loss_graph.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def model_comparison_graph():
    models = ["SVM", "LSTM", "BERT", "Hybrid Transformer"]
    scores = [82, 86, 90, 95]

    plt.figure()
    plt.bar(models, scores)
    plt.title("Model Comparison")
    plt.xlabel("Models")
    plt.ylabel("Accuracy (%)")

    path = os.path.join(CHART_DIR, "model_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def generate_confusion_matrix():
    y_true = [0, 1, 0, 1, 1, 0, 1, 0]
    y_pred = [0, 1, 0, 1, 0, 0, 1, 1]

    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    path = os.path.join(CHART_DIR, "confusion_matrix.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def generate_roc_curve():
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_scores = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.85, 0.4])

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    path = os.path.join(CHART_DIR, "roc_curve.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def generate_wordcloud(text):
    if not text.strip():
        return None

    wc = WordCloud(width=800, height=400, background_color="white").generate(text)

    path = os.path.join(CHART_DIR, "wordcloud.png")
    wc.to_file(path)
    return path