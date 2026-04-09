import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix

from transformers import DistilBertTokenizer

from preprocessing.text_cleaning import clean_text
from features.linguistic_features import extract_features
from model.hybrid_model import HybridModel


# ---------------- DATASET ---------------- #

class TextDataset(Dataset):

    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        text = self.texts[idx]

        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        ling = extract_features(text)

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "ling": torch.tensor(ling, dtype=torch.float),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ---------------- LOAD DATA ---------------- #

df = pd.read_csv("data/dataset.csv")

df["text"] = df["text"].astype(str).apply(clean_text)

texts = df["text"].tolist()
labels = df["label"].tolist()


# ---------------- TOKENIZER ---------------- #

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


dataset = TextDataset(texts, labels, tokenizer)

loader = DataLoader(dataset, batch_size=32)


# ---------------- LOAD MODEL ---------------- #

model = HybridModel()

model.load_state_dict(torch.load("model.pt"))

model.eval()


# ---------------- PREDICTIONS ---------------- #

predictions = []
true_labels = []

with torch.no_grad():

    for batch in loader:

        outputs = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["ling"]
        )

        preds = torch.argmax(outputs, dim=1)

        predictions.extend(preds.tolist())

        true_labels.extend(batch["label"].tolist())


# ---------------- CONFUSION MATRIX ---------------- #

cm = confusion_matrix(true_labels, predictions)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
print("Confusion matrix saved!")