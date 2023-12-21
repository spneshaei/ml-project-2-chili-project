# Description: Helper functions for the DNN training code.

# Import libraries

import json
from collections import Counter
import re
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix

# *******************************************************************************************************************************
# Text Embedding (Only used in early experiments; we did not encode texts in the final experiments due to the computational costs)
# *******************************************************************************************************************************

class CBOWEmbedder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, device='cpu'):
        super(CBOWEmbedder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim).to(device)

    def forward(self, input):
        return self.embeddings(input).mean(dim=0)
    

def create_vocab(text):
    """
    Create a vocabulary from the provided text.

    Args:
        text: The string of text.

    Returns:
        word_to_ix: Dictionary mapping words to indices.
    """

    words = re.findall(r'\w+', text.lower())
    vocab = Counter(words)
    word_to_ix = {word: i for i, (word, _) in enumerate(vocab.items())}
    return word_to_ix


def embed_text(text, device, size=10):
    """
    Embed the provided text using CBOW.

    Args:
        text: String of text.
        device: Device to use for embedding (e.g. `cuda`).
        size: Embedding dimension. The default value is 10.

    Returns:
        Embedding of the text.
    """

    vocab = create_vocab(text)
    embedder = CBOWEmbedder(len(vocab), size).to(device)
    words = re.findall(r'\w+', text.lower())
    indices = torch.tensor([vocab[word] for word in words if word in vocab], dtype=torch.long).to(device)
    return embedder(indices)



# *******************************************************************************************************************************
# Data Loading and Processing
# In the final experiments, we only extract difficulty and previous students answers from the data (and, additionally, the KCs), due to computational costs.
# Prior to that, we experimented with different data attributes (e.g., question text, KC text, etc.)
# *******************************************************************************************************************************

def load_and_process_data(student_filename, seq_len=10):
    """
    Load the data and extract the relevant attributes.
    We choose only difficulty and previous students answers in our final experiments.

    Args:
        student_filename: Path to the student data file.
        seq_len: Length of the sequence in the provided student data. The default value is 10.

    Returns:
        full_data: List of lists of dictionaries, where each dictionary contains the relevant attributes for each question in the sequence.
    """

    # Load the data
    with open(student_filename) as f:
        student_data = json.load(f)

    # Extract and data
    full_data = []
    for student in tqdm(student_data):
        data = []
        for i in range(seq_len):
            q_id = int(student['question_ids'][i])
            data.append({
                'difficulty': int(student['difficulties'][i]),
                'answer': int(student['answers'][i])
            })
        full_data.append(data)
    
    return full_data


class StudentSequenceDataset(Dataset):
    """
    Dataset class for the student data.
    We only provide this for compatibility with the PyTorch DataLoader.
    """

    def __init__(self, data, embedding_dim):
        self.data = data
        self.embedding_dim = embedding_dim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        input_sequence = []
        for i in range(len(sequence)):
            element = sequence[i]
            features = [
                torch.tensor([element['difficulty']], dtype=torch.float32).cpu().detach()
            ]
            if i < len(sequence) - 1:
                features.append(torch.tensor([element['answer']], dtype=torch.float32).cpu().detach())
            else:
                features.append(torch.zeros(1, dtype=torch.float32).cpu().detach())  # Padding for the last element, because we have no answer for it
            concatenated = torch.cat(features)
            input_sequence.append(concatenated)
        label = torch.tensor([sequence[-1]['answer']], dtype=torch.float32).cpu().detach()
        return torch.stack(input_sequence), label
    



# *******************************************************************************************************************************
# Sequence Models (RNN, LSTM, and Transformer).
# All models used a sigmoid activation function in the final layer. This helps provide normalized confidence scores.
# *******************************************************************************************************************************

# RNN Model
class SimpleRNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=50, num_layers=1, device='cpu'):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True).to(device)
        self.fc = nn.Linear(50, 1).to(device)

    def forward(self, x):
        out, _ = self.rnn(x)
        return torch.sigmoid(self.fc(out[:, -1, :]))
    
# LSTM Model
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=50, num_layers=1, device='cpu'):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True).to(device)
        self.fc = nn.Linear(hidden_size, 1).to(device)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        return torch.sigmoid(self.fc(out[:, -1, :]))

# Transformer Model
class SimpleTransformer(nn.Module):
    def __init__(self, input_size=2, d_model=64, nhead=8, num_layers=1, dropout=0.1, device='cpu'):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(0, 1)
        transformer_output = self.transformer_encoder(x)
        out = transformer_output[-1]
        return torch.sigmoid(self.fc(out))


# *******************************************************************************************************************************
# Training and Evaluation Functions
# *******************************************************************************************************************************

def train(model, train_data_loader, val_data_loader, test_data_loader, device, epochs=10, lr=0.001):
    """
    Train the model and evaluate on the validation and test sets after each epoch.

    Args:
        model: The model to train (object of type SimpleRNN, SimpleLSTM, or SimpleTransformer).
        train_data_loader: PyTorch DataLoader for the training set.
        val_data_loader: PyTorch DataLoader for the validation set.
        test_data_loader: PyTorch DataLoader for the test set.
        device: Device to use for training and evaluation (e.g. `cuda`).
        epochs: Number of epochs to train for. The default value is 10.
        lr: Initial learning rate for the Adam optimizer. The default value is 0.001.
    """

    criterion = nn.BCEWithLogitsLoss() # Per PyTorch docs (https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html): "This loss combines a Sigmoid layer and the BCELoss in one single class. This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability."
    
    optimizer = optim.Adam(model.parameters(), lr=lr) # make the optimizer

    # Train loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        loop = tqdm(train_data_loader, leave=True, desc=f"Epoch [{epoch+1}/{epochs}]")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=total_loss / (loop.n + 1))

        # Evaluate after each epoch
        val_acc, val_balanced_acc, val_f1_0, val_f1_1, val_support_0, val_support_1, val_auc = evaluate(model, val_data_loader, device)

        test_acc, test_balanced_acc, test_f1_0, test_f1_1, test_support_0, test_support_1, test_auc = evaluate(model, test_data_loader, device)
        
        return val_acc, val_balanced_acc, val_f1_0, val_f1_1, val_support_0, val_support_1, val_auc, test_acc, test_balanced_acc, test_f1_0, test_f1_1, test_support_0, test_support_1, test_auc
    

def evaluate(model, data_loader, device):
    """
    Evaluate the model on the provided data.

    Args:
        model: the model to train (object of type SimpleRNN, SimpleLSTM, or SimpleTransformer).
        data_loader: PyTorch DataLoader for the data.
        device: Device to use for evaluation (e.g. `cuda`).
    
    Returns:
        acc: Accuracy.
        balanced_acc: Balanced accuracy.
        f1_0: F1 score for class 0 (wrong answer).
        f1_1: F1 score for class 1 (correct answer).
        support_0: Support for class 0 (wrong answer).
        support_1: Support for class 1 (correct answer).
        auc: Area under the ROC curve (AUC).
    """

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs >= 0.5).float()
            all_preds.extend(predicted.view(-1).tolist())
            all_labels.extend(labels.view(-1).tolist())

    # Find the scores from all_preds and all_labels
    acc = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    f1_scores = f1_score(all_labels, all_preds, average=None)
    supports = confusion_matrix(all_labels, all_preds).sum(axis=1)

    return acc, balanced_acc, f1_scores[0], f1_scores[1], supports[0], supports[1], auc


def predict(model, data_loader, device, with_confidence=False):
    """
    Predict the model on the provided data.
    While similar to evaluate(), this function returns the predctions, labels, and (optionally) confidences instead of the metrics.

    Args:
        model: The model to train (object of type SimpleRNN, SimpleLSTM, or SimpleTransformer).
        data_loader: PyTorch DataLoader for the data.
        device: Device to use for evaluation (e.g. `cuda`).
        with_confidence: Whether to return the confidence scores or not. The default value is False.

    Returns:
        all_labels: The list of labels.
        all_preds: The list of predictions.
        all_confs: The list of confidence scores (if with_confidence is True).
    """
    model.eval()
    all_preds, all_labels = [], []
    if with_confidence:
        all_confs = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs >= 0.5).float() # threshold of 0.5
            if with_confidence:
                all_confs.extend(outputs.view(-1).tolist())
            all_preds.extend(predicted.view(-1).tolist())
            all_labels.extend(labels.view(-1).tolist())
            
    return all_labels, all_preds, all_confs if with_confidence else None



# *******************************************************************************************************************************
# Training and Evaluation Object
# *******************************************************************************************************************************

class ModelEvaluator:
    """
    Class for training and evaluating multiple models (useful for grid search).

    Args:
        models: Dictionary of models to train and evaluate.
        train_data_loader: PyTorch DataLoader for the training set.
        val_data_loader: PyTorch DataLoader for the validation set.
        test_data_loader: PyTorch DataLoader for the test set.
        device: Device to use for training and evaluation (e.g. `cuda`).
        epochs: Number of epochs to train for. The default value is 100.
        lr: Initial learning rate. The default value is 0.001.
    """

    def __init__(self, models, train_data_loader, val_data_loader, test_data_loader, device, epochs=100, lr=0.001):
        self.models = models
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.all_val_metrics = {model_name: {'acc': 0, 'bal_acc': 0, 'f1_0': 0, 'f1_1': 0, 'sup_0': 0, 'sup_1': 0, 'auc': 0, 'epoch': 0} for model_name in models}
        self.all_test_metrics = {model_name: {'acc': 0, 'bal_acc': 0, 'f1_0': 0, 'f1_1': 0, 'sup_0': 0, 'sup_1': 0, 'auc': 0, 'epoch': 0} for model_name in models}

    def train_and_evaluate(self):
        """
        Train and evaluate the provided models on the validation and test sets.
        """

        for model_name, model in self.models.items():
            with tqdm(total=self.epochs) as pbar:
                for epoch in range(self.epochs):
                    val_acc, val_balanced_acc, val_f1_0, val_f1_1, val_support_0, val_support_1, val_auc, test_acc, test_balanced_acc, test_f1_0, test_f1_1, test_support_0, test_support_1, test_auc = train(model, self.train_data_loader, self.val_data_loader, self.test_data_loader, self.device, epochs=1, lr=self.lr)
                    self.all_val_metrics[f'{model_name}_{epoch}'] = {'acc': val_acc, 'bal_acc': val_balanced_acc, 'f1_0': val_f1_0, 'f1_1': val_f1_1, 'sup_0': val_support_0, 'sup_1': val_support_1, 'auc': val_auc, 'epoch': epoch}
                    self.all_test_metrics[f'{model_name}_{epoch}'] = {'acc': test_acc, 'bal_acc': test_balanced_acc, 'f1_0': test_f1_0, 'f1_1': test_f1_1, 'sup_0': test_support_0, 'sup_1': test_support_1, 'auc': test_auc, 'epoch': epoch}
                    pbar.update(1)
                
    
    def compute_best_metrics(self, print_dict=True, save_dict=True):
        """
        Compute the best metrics (based on balanced accuracy) on the validation and test sets.
        The "best" model across each grid configuration is determined using the highest VALIDATION balanced accuracy.
        The validation and test metrics can all be dumped as JSON files.

        Args:
            print_dict: Whether to print the best metrics or not. The default value is True.
            save_dict: Whether to save the metrics or not. The default value is True.

        Returns:
            best_val_metrics: The dictionary of the best validation metrics.
            best_test_metrics: The dictionary of the best test metrics.
        """

        class NpEncoder(json.JSONEncoder): # to fix the "Object of type 'int64' is not JSON serializable" error: Source: https://stackoverflow.com/a/57915246
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)
            
        best_val_balanced_acc = 0
        best_val_model_name = None
        for model_name, metrics in self.all_val_metrics.items(): # loop to find the best
            if metrics['bal_acc'] > best_val_balanced_acc:
                best_val_balanced_acc = metrics['bal_acc']
                best_val_model_name = model_name
        
        best_test_balanced_acc = 0
        best_test_model_name = None
        for model_name, metrics in self.all_test_metrics.items(): # loop to find the best
            if metrics['bal_acc'] > best_test_balanced_acc:
                best_test_balanced_acc = metrics['bal_acc']
                best_test_model_name = model_name

        if print_dict:
            print("Best Metrics:")
            print(f"    Validation Set: {best_val_model_name}: {self.all_val_metrics[best_val_model_name]}")
            print(f"    Test Set: {best_test_model_name}: {self.all_test_metrics[best_test_model_name]}\n")

        if save_dict:
            # Save the best metrics
            with open(f'metrics/best_{best_val_model_name}_val.json', 'w') as f:
                json.dump(self.all_val_metrics[best_val_model_name], f, cls=NpEncoder)
            with open(f'metrics/best_{best_test_model_name}_test.json', 'w') as f:
                json.dump(self.all_test_metrics[best_test_model_name], f, cls=NpEncoder)

            # Save all metrics
            with open(f'metrics/all_{best_val_model_name}_val.json', 'w') as f:
                json.dump(self.all_val_metrics, f, cls=NpEncoder)
            with open(f'metrics/all_{best_test_model_name}_test.json', 'w') as f:
                json.dump(self.all_test_metrics, f, cls=NpEncoder)

        return self.all_val_metrics[best_val_model_name], self.all_test_metrics[best_test_model_name]
    
    # Temporary wrapper function for demo purposes (confidence scores). For proper training/evaluation, use train_and_evaluate().
    def predict(self, model_name, data_loader, device, with_confidence=False):
        model = self.models[model_name]
        return predict(model, data_loader, device, with_confidence=with_confidence)
