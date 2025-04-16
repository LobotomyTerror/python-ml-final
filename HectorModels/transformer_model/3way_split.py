import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("Data/Cleaned_Symptom2DiseaseGroup.csv")

# Encode labels
label_encoder = LabelEncoder()
df["label_id"] = label_encoder.fit_transform(df["label"])

# 1. Sample 2 examples per class for test set
test_df = df.groupby("label").sample(n=2, random_state=42)
remaining_df = df.drop(test_df.index)

# 2. Stratified split for train/val from the remaining data
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(splitter.split(remaining_df["clean_text"], remaining_df["label_id"]))
train_df = remaining_df.iloc[train_idx]
val_df = remaining_df.iloc[val_idx]

# 3. Save outputs
train_df.to_csv("Data/train.csv", index=False)
val_df.to_csv("Data/validation.csv", index=False)
test_df.to_csv("Data/test.csv", index=False)

# 4. Print label counts
print("\n✅ SPLIT COMPLETED:")
print("\nTrain label counts:\n", train_df["label"].value_counts())
print("\nValidation label counts:\n", val_df["label"].value_counts())
print("\nTest label counts (should be 2 each):\n", test_df["label"].value_counts())
