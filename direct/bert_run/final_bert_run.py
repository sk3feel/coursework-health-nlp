import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification


model_dir = "../bert_training/final_training/bert_disease_model"
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_df = pd.read_csv("../data/final_data/train_augmented.csv")
le.fit(train_df['disease'])

new_df = pd.read_csv("./new_symptoms.csv")

assert 'symptoms' in new_df.columns, "В файле должна быть колонка 'symptoms'"

texts = new_df['symptoms'].tolist()

encodings = tokenizer(
    texts,
    truncation=True,
    padding='max_length',
    max_length=128,
    return_tensors='pt'
)

input_ids = encodings['input_ids'].to(device)
attention_mask = encodings['attention_mask'].to(device)

with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    predictions = torch.argmax(outputs.logits, dim=1)

predicted_diseases = le.inverse_transform(predictions.cpu().numpy())

new_df['predicted_disease'] = predicted_diseases
print(new_df[['symptoms', 'predicted_disease']])

output_path = "final_predicted.csv"
new_df.to_csv(output_path, index=False)
print(f"\nРезультаты сохранены в {output_path}")
