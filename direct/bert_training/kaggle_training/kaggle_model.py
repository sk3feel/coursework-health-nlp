import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm

train_df = pd.read_csv("../../data/kaggle_data/train_augmented.csv")
test_df = pd.read_csv("../../data/kaggle_data/test.csv")

le = LabelEncoder()
train_df['label'] = le.fit_transform(train_df['disease'])
test_df['label'] = le.transform(test_df['Disease'])

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class SymptomDataset(Dataset):
    def __init__(self, df):
        self.texts = df['symptoms'].tolist()
        self.labels = df['label'].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx])
        }

train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['label'])

train_dataset = SymptomDataset(train_df)
val_dataset = SymptomDataset(val_df)
test_dataset = SymptomDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = len(le.classes_)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

train_loss_hist = []
train_acc_hist = []
train_f1_hist = []

best_f1 = 0
patience = 2
patience_counter = 0
best_model_state = None

for epoch in range(20):  # максимум 20 эпох
    model.train()
    total_loss = 0
    true_labels = []
    pred_labels = []

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)

        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')

    train_loss_hist.append(avg_loss)
    train_acc_hist.append(acc)
    train_f1_hist.append(f1)

    print(f"\nEpoch {epoch+1}: Train loss = {avg_loss:.4f}, acc = {acc:.4f}, f1 = {f1:.4f}")

    model.eval()
    val_true = []
    val_pred = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            val_true.extend(labels.cpu().numpy())
            val_pred.extend(preds.cpu().numpy())

    val_f1 = f1_score(val_true, val_pred, average='weighted')
    print(f"Validation F1: {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        patience_counter = 0
        best_model_state = model.state_dict()
        print("новая модель лучшая.")
    else:
        patience_counter += 1
        print(f"улучшений нет. ждем: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print(" Early stopping.")
            break

if best_model_state:
    model.load_state_dict(best_model_state)

model.eval()
true_test, pred_test = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        true_test.extend(labels.cpu().numpy())
        pred_test.extend(preds.cpu().numpy())

test_acc = accuracy_score(true_test, pred_test)
test_f1 = f1_score(true_test, pred_test, average='weighted')
print("\n--- Test Results ---")
print(f"Accuracy: {test_acc:.4f}")
print(f"F1 Score: {test_f1:.4f}")
print(classification_report(true_test, pred_test, target_names=le.classes_))

plot_dir = "./plots"
os.makedirs(plot_dir, exist_ok=True)

plt.figure()
plt.plot(train_loss_hist, label="Loss")
plt.title("Loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig(os.path.join(plot_dir, "loss.png"))
plt.close()

plt.figure()
plt.plot(train_acc_hist, label="Accuracy")
plt.title("Accuracy over epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.savefig(os.path.join(plot_dir, "accuracy.png"))
plt.close()

plt.figure()
plt.plot(train_f1_hist, label="F1 Score")
plt.title("F1 Score over epochs")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.legend()
plt.grid()
plt.savefig(os.path.join(plot_dir, "f1_score.png"))
plt.close()

print(f"\n Графики сохранены")

output_dir = "./bert_disease_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"\n Модель и токенизатор: {output_dir}")

