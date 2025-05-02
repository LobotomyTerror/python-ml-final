import pandas as pd

# Load both datasets
processed_df = pd.read_csv("Data/Processed_Symptom2DiseaseGroup_Labeled.csv")
cleaned_df = pd.read_csv("Data/Cleaned_Symptom2DiseaseGroup.csv")

# Ensure they have the same number of rows
if len(processed_df) != len(cleaned_df):
    raise ValueError("Files do not have the same number of rows. Cannot match by index.")

# Replace label_name in processed with label from cleaned
processed_df["label_name"] = cleaned_df["label"]

# Save updated version
output_path = "Data/Processed_Symptom2DiseaseGroup_Labeled_Updated.csv"
processed_df.to_csv(output_path, index=False)
print(f"✅ Updated file saved to: {output_path}")
