import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

train_path = "../../data/final_data/train_augmented.csv"
model_dir = "./bert_disease_model"

df = pd.read_csv(train_path)

le = LabelEncoder()
le.fit(df['disease'])

joblib.dump(le, os.path.join(model_dir, "label_encoder.joblib"))
print("успех")
