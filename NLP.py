import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load the data
train_df = pd.read_csv('/path/to/train.csv')
test_df = pd.read_csv('/path/to/test.csv')

# Drop rows with NaN values in necessary columns
train_df = train_df.dropna(subset=['crimeaditionalinfo', 'category', 'sub_category'])
test_df = test_df.dropna(subset=['crimeaditionalinfo', 'category', 'sub_category'])

# Encode labels
train_df['label'] = train_df['category'].astype('category').cat.codes
test_df['label'] = test_df['category'].astype('category').cat.codes

# Check and handle NaN values in the test labels
if test_df['label'].isnull().any():
    print("NaN values found in test labels. Removing rows with NaN labels.")
    test_df = test_df.dropna(subset=['label']).reset_index(drop=True)

# Convert labels to integers
train_df['label'] = train_df['label'].astype(int)
test_df['label'] = test_df['label'].astype(int)

# Define a custom dataset class
class CrimeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = int(self.labels[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(train_df['label'].unique()))

# Prepare datasets
train_texts, val_texts, train_labels, val_labels = train_test_split(train_df['crimeaditionalinfo'], train_df['label'], test_size=0.2)
train_dataset = CrimeDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
val_dataset = CrimeDataset(val_texts.tolist(), val_labels.tolist(), tokenizer)
test_dataset = CrimeDataset(test_df['crimeaditionalinfo'].tolist(), test_df['label'].tolist(), tokenizer)

# Define metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model on the validation set
trainer.evaluate()

# Make predictions on the test set
test_preds = trainer.predict(test_dataset)
test_pred_labels = test_preds.predictions.argmax(-1)

# Optionally, save predictions in a DataFrame
test_df['predicted_label'] = test_pred_labels
test_df.to_csv('test_predictions.csv', index=False)
print("Predictions saved to 'test_predictions.csv'")
