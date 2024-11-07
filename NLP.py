import pandas as pd
import re
from google.colab import drive
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

# Mount Google Drive
drive.mount('/content/drive')

# Load datasets from Google Drive
train_df = pd.read_csv('/content/drive/MyDrive/train.csv')
test_df = pd.read_csv('/content/drive/MyDrive/test.csv')

# Combine 'sub_category' and 'crimeaditionalinfo' into a single text column
train_df['text'] = train_df['sub_category'].astype(str) + ' ' + train_df['crimeaditionalinfo'].astype(str)
test_df['text'] = test_df['sub_category'].astype(str) + ' ' + test_df['crimeaditionalinfo'].astype(str)

# Define Dataset class for tokenization and label encoding
class CyberDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = int(self.labels[idx])  # Ensure label is converted to int
        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)  # Ensure int64 tensor
        }

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=train_df['category'].nunique())

# Encode labels
label_mapping = {label: idx for idx, label in enumerate(train_df['category'].unique())}
train_df['label'] = train_df['category'].map(label_mapping)
test_df['label'] = test_df['category'].map(label_mapping)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(train_df['text'], train_df['label'], test_size=0.2, random_state=42)

# Create training, validation, and test datasets
train_dataset = CyberDataset(X_train.tolist(), y_train.tolist(), tokenizer)
val_dataset = CyberDataset(X_val.tolist(), y_val.tolist(), tokenizer)
test_dataset = CyberDataset(test_df['text'].tolist(), test_df['label'].tolist(), tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted', zero_division=1)
    recall = recall_score(labels, predictions, average='weighted', zero_division=1)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=1)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate on validation set
val_metrics = trainer.evaluate(eval_dataset=val_dataset)
print(f"Validation Metrics: {val_metrics}")

# Make predictions on the test set
test_preds = trainer.predict(test_dataset)
test_pred_labels = test_preds.predictions.argmax(-1)

# Map predictions back to original labels
inverse_label_mapping = {v: k for k, v in label_mapping.items()}
test_pred_categories = [inverse_label_mapping[label] for label in test_pred_labels]

# Print classification report on test set
print("Classification Report (Test):")
print(classification_report(test_df['label'], test_pred_labels, target_names=list(label_mapping.keys())))
