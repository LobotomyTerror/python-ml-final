import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score

# Load pre-split datasets
train_df = pd.read_csv("Data/train.csv")
val_df = pd.read_csv("Data/validation.csv")
test_df = pd.read_csv("Data/test.csv")

# Encode labels
label_encoder = LabelEncoder()
train_df["label_id"] = label_encoder.fit_transform(train_df["label"])
val_df["label_id"] = label_encoder.transform(val_df["label"])
test_df["label_id"] = label_encoder.transform(test_df["label"])

# Convert to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df[["clean_text", "label_id"]])
val_dataset = Dataset.from_pandas(val_df[["clean_text", "label_id"]])
test_dataset = Dataset.from_pandas(test_df[["clean_text", "label_id"]])

# Load ClinicalBERT
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["clean_text"], truncation=True, padding=True, max_length=512)

# Tokenize all sets
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Rename labels column to "label"
train_dataset = train_dataset.rename_column("label_id", "label")
val_dataset = val_dataset.rename_column("label_id", "label")
test_dataset = test_dataset.rename_column("label_id", "label")

# Set format for PyTorch
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Training arguments
training_args = TrainingArguments(
    output_dir="./clinicalbert_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Evaluation metric
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate on the test set
test_results = trainer.evaluate(eval_dataset=test_dataset)
print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")

# Save label mapping
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
pd.Series(label_mapping).to_csv("label_mapping.csv")
