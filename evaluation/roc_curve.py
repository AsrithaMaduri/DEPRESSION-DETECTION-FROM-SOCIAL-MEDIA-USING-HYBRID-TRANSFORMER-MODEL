import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import torch
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from torch.utils.data import Dataset, DataLoader

from transformers import DistilBertTokenizer

from preprocessing.text_cleaning import clean_text
from features.linguistic_features import extract_features
from model.hybrid_model import HybridModel


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


df = pd.read_csv("data/dataset.csv")

df["text"] = df["text"].apply(clean_text)

texts = df["text"].tolist()
labels = df["label"].tolist()

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

dataset = TextDataset(texts, labels, tokenizer)

loader = DataLoader(dataset, batch_size=32)

model = HybridModel()

model.load_state_dict(torch.load("model.pt"))

model.eval()

y_true = []
y_scores = []

with torch.no_grad():

    for batch in loader:

        outputs = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["ling"]
        )

        probs = torch.softmax(outputs, dim=1)[:,1]

        y_scores.extend(probs.tolist())
        y_true.extend(batch["label"].tolist())


fpr, tpr, _ = roc_curve(y_true, y_scores)

roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)

plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend(loc="lower right")

plt.show()