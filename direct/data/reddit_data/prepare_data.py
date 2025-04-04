import pandas as pd
import random
from sklearn.model_selection import train_test_split

df = pd.read_csv("reddit_dataset.csv")

symptom_cols = [col for col in df.columns if "symptom" in col.lower()]

df["symptoms"] = df[symptom_cols].astype(str).apply(
    lambda row: ", ".join([s.strip() for s in row if s.strip().lower() != 'nan' and s.strip() != '']),
    axis=1
)

df = df[df["symptoms"].apply(lambda s: len([x for x in s.split(",") if x.strip()]) >= 2)]

train_df, test_df = train_test_split(
    df[["symptoms", "Disease"]],
    test_size=0.2,
    random_state=42,
    stratify=df["Disease"]
)

def augment_symptoms(symptom_text, n_versions=3):
    symptoms = [s.strip() for s in symptom_text.split(",") if s.strip()]
    symptoms = list(set(symptoms))
    if len(symptoms) < 2:
        return [", ".join(symptoms)]
    versions = set()
    attempts = 0
    while len(versions) < n_versions and attempts < 10 * n_versions:
        random.shuffle(symptoms)
        versions.add(", ".join(symptoms))
        attempts += 1
    return list(versions)

augmented_data = []
for _, row in train_df.iterrows():
    for aug in augment_symptoms(row["symptoms"], n_versions=3):
        augmented_data.append({"symptoms": aug, "disease": row["Disease"]})

train_aug_df = pd.DataFrame(augmented_data)

train_aug_df.to_csv("train_augmented.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("✅ Успешно сохранены файлы: train_augmented.csv и test.csv")
