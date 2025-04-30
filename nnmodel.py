import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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
        self.layer1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.layer2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.layer3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.layer4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)

        self.layer5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)

        self.layer6 = nn.Linear(64, 32)
        self.bn6 = nn.BatchNorm1d(32)

        self.layer7 = nn.Linear(32, 1)

        self.act = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dropout(self.act(self.bn1(self.layer1(x))))
        x = self.dropout(self.act(self.bn2(self.layer2(x))))
        x = self.dropout(self.act(self.bn3(self.layer3(x))))
        x = self.dropout(self.act(self.bn4(self.layer4(x))))
        x = self.dropout(self.act(self.bn5(self.layer5(x))))
        x = self.dropout(self.act(self.bn6(self.layer6(x))))
        x = self.layer7(x)
        return x


def train_model(X_train_tensor, y_train_tensor, input_dim, epochs=100, batch_size=32):
    train_dataset = HorseRaceDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = HorseRacePredictor(input_dim)

    num_pos = (y_train_tensor == 1).sum().item()
    num_neg = (y_train_tensor == 0).sum().item()
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

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

        avg_loss = running_loss / len(train_loader)
        scheduler.step(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return model


def test_model(model, X_test, y_test):
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test)

    with torch.no_grad():
        logits = model(X_test_tensor).squeeze()
        probs = torch.sigmoid(logits)
        predictions = (probs >= 0.5).float().numpy()

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='binary', pos_label=1)
    recall = recall_score(y_test, predictions, average='binary', pos_label=1)
    f1 = f1_score(y_test, predictions, average='binary', pos_label=1)
    conf_matrix = confusion_matrix(y_test, predictions)


    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'predictions': predictions
    }


def run_neural_network(X_train, y_train, X_test, y_test):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    y_train = y_train.astype(np.float32).values
    y_test = y_test.astype(np.float32).values

    input_dim = X_train.shape[1]
    model = train_model(X_train_scaled, y_train, input_dim)
    results = test_model(model, X_test_scaled, y_test)
    return model, results
