"""
nnmodel.py - Implementation of Neural Network Model for Horse Race Prediction

This module provides functions to train and test a neural network model
on horse race datasets.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class HorseRaceDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class HorseRacePredictor(nn.Module):
    def __init__(self, input_dim):
        super(HorseRacePredictor, self).__init__()
        # Simpler architecture
        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer4(x)
        return x

def train_model(X_train_tensor, y_train_tensor, input_dim, epochs=100, batch_size=32):
    train_dataset = HorseRaceDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = HorseRacePredictor(input_dim)

    # calculate class imbalance ratio
    num_pos = np.sum(y_train_tensor)
    num_neg = len(y_train_tensor) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)

    # use BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")
    
    return model


def test_model(model, X_test, y_test):
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test)

    with torch.no_grad():
        logits = model(X_test_tensor).squeeze()
        probs = torch.sigmoid(logits)
        predictions = (probs >= 0.5).float().numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    conf_matrix = confusion_matrix(y_test, predictions)
    
    # Return results as a dictionary
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'predictions': predictions
    }
    
    return results

def run_neural_network(X_train, y_train, X_test, y_test):
    # Feature scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    # Label cleanup
    y_train = y_train.astype(np.float32).values
    y_test = y_test.astype(np.float32).values

    # Input dimension
    input_dim = X_train.shape[1]

    # Train and test
    model = train_model(X_train_scaled, y_train, input_dim)
    results = test_model(model, X_test_scaled, y_test)

    return model, results
