import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from preprocessing.text_cleaning import clean_text
from features.linguistic_features import extract_features
from model.hybrid_model import HybridModel


# ---------------- DEVICE ---------------- #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------- DATASET CLASS ---------------- #

class TextDataset(Dataset):

    def __init__(self, texts, labels, tokenizer):

        self.labels = labels

        self.encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=64
        )

        self.ling_features = [extract_features(t) for t in texts]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        item["ling"] = torch.tensor(self.ling_features[idx], dtype=torch.float)
        item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


# ---------------- LOAD DATASETS ---------------- #

df1 = pd.read_csv("data/dataset.csv")
df2 = pd.read_csv("data/Mental-Health-Twitter.csv")

df2 = df2[["post_text", "label"]]
df2 = df2.rename(columns={"post_text": "text"})

df = pd.concat([df1, df2], ignore_index=True)

print("Total samples:", len(df))


# ---------------- CLEAN TEXT ---------------- #

df["text"] = df["text"].astype(str).apply(clean_text)

# remove duplicate texts
df = df.drop_duplicates(subset="text")


# ---------------- TRAIN TEST SPLIT ---------------- #

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print("Train size:", len(train_df))
print("Test size:", len(test_df))


# ---------------- TOKENIZER ---------------- #

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


# ---------------- DATASETS ---------------- #

train_ds = TextDataset(
    train_df.text.tolist(),
    train_df.label.tolist(),
    tokenizer
)

test_ds = TextDataset(
    test_df.text.tolist(),
    test_df.label.tolist(),
    tokenizer
)


# ---------------- DATALOADERS ---------------- #

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)


# ---------------- MODEL ---------------- #

model = HybridModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()


# ---------------- TRAINING LOOP ---------------- #

EPOCHS = 3

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}")

    model.train()

    total_loss = 0

    for batch in train_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        ling = batch["ling"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            input_ids,
            attention_mask,
            ling
        )

        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Loss:", total_loss)


    # -------- TRAINING ACCURACY -------- #

    model.eval()

    train_preds = []
    train_labels = []

    with torch.no_grad():

        for batch in train_loader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ling = batch["ling"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask, ling)

            preds = torch.argmax(outputs, dim=1)

            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

    train_acc = accuracy_score(train_labels, train_preds)

    print("Training Accuracy:", train_acc)


# ---------------- TEST EVALUATION ---------------- #

print("\nEvaluating on test set...")

model.eval()

predictions = []
true_labels = []

with torch.no_grad():

    for batch in test_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        ling = batch["ling"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask, ling)

        preds = torch.argmax(outputs, dim=1)

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())


test_acc = accuracy_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

print("\nTest Accuracy:", test_acc)
print("F1 Score:", f1)


# ---------------- SAVE MODEL ---------------- #

torch.save(model.state_dict(), "model.pt")

print("\nModel saved successfully!")