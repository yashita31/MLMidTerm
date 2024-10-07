import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Step 1: Load the CSV files
def load_data(file1, file2, file3, file4, file5):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    df4 = pd.read_csv(file4)
    df5 = pd.read_csv(file5)

    # Add a column to identify which model each sentence came from
    df1['Model'] = 'LLM1'
    df2['Model'] = 'LLM2'
    df3['Model'] = 'LLM3'
    df4['Model'] = 'LLM4'
    df5['Model'] = 'LLM5'

    # Combine the dataframes
    df_combined = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
    df_combined['combined_text'] = df_combined.iloc[:, 0] + " " + df_combined.iloc[:, 1]

    return df_combined

# Custom Dataset
class TextDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, model):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.model = model

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]

        # Ensure sentence is a string
        if not isinstance(sentence, str):
            sentence = str(sentence)

        return {
            'sentence': sentence,
            'labels': torch.tensor(label, dtype=torch.long)
        }

def collate_fn(batch):
    sentences = [item['sentence'] for item in batch]
    labels = torch.tensor([item['labels'] for item in batch])

    # Tokenize the entire batch of sentences
    encoding = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)

    # Move encoding tensors to the appropriate device
    encoding = {k: v.to(device) for k, v in encoding.items()}

    # Get embeddings for the batch of sentences
    with torch.no_grad():
        outputs = sbert_model(**encoding).last_hidden_state
        pooled_output = mean_pooling(outputs, encoding['attention_mask'])

    return {
        'embeddings': pooled_output,  # Batch of embeddings
        'labels': labels
    }

# Function to get sentence embeddings using SBERT
def get_embeddings(sentences, tokenizer, model):
    embeddings = []
    with torch.no_grad():
        encoding = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
        encoding = {k: v.to(model.device) for k, v in encoding.items() }
        outputs = model(**encoding).last_hidden_state.mean(dim=1)
        pooled_output = mean_pooling(outputs, encoding['attention_mask'])
        embeddings.append(pooled_output.cpu())
    return torch.vstack(embeddings)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(-1, -1, token_embeddings.size(-1)).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Simple fully connected network
class SimpleFCClassifier(nn.Module):
    def __init__(self, input_dim, n_classes, drop_out=0.3, hidden_layers=128):
        super(SimpleFCClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_layers)
        self.fc2 = nn.Linear(hidden_layers, n_classes)
        self.drop = nn.Dropout(p=drop_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

# Training function
def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    losses = []
    for batch_idx, data in enumerate(data_loader):
        embeddings = data['embeddings'].to(device)
        labels = data['labels'].to(device)

        # Forward pass
        outputs = model(embeddings)
        loss = loss_fn(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return sum(losses)/len(losses)

# Function to evaluate model on test data
def test_epoch(model, data_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in data_loader:
            embeddings = batch['embeddings'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(embeddings)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

# Save the model and optimizer checkpoint
def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))

# Load model and optimizer checkpoint
def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Loaded model from epoch {epoch}")
        return model, optimizer, epoch
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
        return model, optimizer, 0

if __name__ == '__main__':
    # Hyperparameters
    num_epochs = 15
    checkpoint_interval = 5
    learning_rate = 0.001

    # Load data
    file1 = 'data/gpt-neo-4k.csv'
    file2 = 'data/fb-opt-4k.csv'
    file3 = 'data/gpt2-4k.csv'
    file4 = 'data/gemma_modified.csv'
    file5 = 'data/mistral_openorca_4k.csv'

    df_combined = load_data(file1, file2, file3, file4, file5)

    # Encode labels
    le = LabelEncoder()
    df_combined['label'] = le.fit_transform(df_combined['Model'])

    # Train/test split
    train_sentences, test_sentences, train_labels, test_labels = train_test_split(
        df_combined['combined_text'], df_combined['label'], test_size=0.2, random_state=42
    )

    # Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    sbert_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(device)

    # Create datasets and dataloaders
    train_dataset = TextDataset(train_sentences.tolist(), train_labels.to_numpy(), tokenizer, sbert_model)
    test_dataset = TextDataset(test_sentences.tolist(), test_labels.to_numpy(), tokenizer, sbert_model)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Initialize model, loss function, and optimizer
    input_dim = 768
    output_dim = 5
    drop_out = 0.5
    hidden_layers = 128
    model = SimpleFCClassifier(input_dim, output_dim, drop_out=drop_out, hidden_layers=hidden_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Check if there is a saved model checkpoint and load it
    checkpoint_dir = 'model_checkpoints'
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_epoch_5.pth')
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

    # Training loop with train and test accuracy printing
    for epoch in range(start_epoch, num_epochs):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate on the test set
        test_accuracy = test_epoch(model, test_loader, device)

        # Evaluate on the train set (optional, to check overfitting)
        train_accuracy = test_epoch(model, train_loader, device)

        # Print epoch results
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

        # Save the model checkpoint every checkpoint_interval epochs
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch, checkpoint_dir)

    # Final evaluation and classification report
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            embeddings = batch['embeddings'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(embeddings)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Calculate accuracy and classification report
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Final Test Accuracy: {accuracy:.4f}")
    report = classification_report(y_true, y_pred)
    print(report)
