import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the correct labels
df = pd.read_csv("Data/Cleaned_Symptom2DiseaseGroup.csv")

# Get unique class names (strings)
classes = sorted(df["label"].unique())

# Create new LabelEncoder using actual names
le = LabelEncoder()
le.fit(classes)

# Save the updated LabelEncoder
with open("Data/label_encoder_correct.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Saved updated LabelEncoder with class names.")
