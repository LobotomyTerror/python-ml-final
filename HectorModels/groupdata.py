import pandas as pd

# Load the CSV
df = pd.read_csv("Data/Symptom2Disease.csv")

# Sort by label (optional, but useful)
df_sorted = df.sort_values(by=["label", "id"])

# Drop the 'id' column
df_sorted = df_sorted.drop(columns=["id"])

# Save the cleaned CSV
df_sorted.to_csv("Data/Symptom2DiseaseGroup.csv", index=False)

print("✔️ 'id' column removed and CSV saved as 'Symptom2DiseaseGroup.csv'")
