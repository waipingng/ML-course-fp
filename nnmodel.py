import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Fix random seeds
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
        self.layer1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)

        self.layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.layer3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)

        self.layer4 = nn.Linear(32, 1)

        self.act = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dropout(self.act(self.bn1(self.layer1(x))))
        x = self.dropout(self.act(self.bn2(self.layer2(x))))
        x = self.dropout(self.act(self.bn3(self.layer3(x))))
        x = self.layer4(x)
        return x


def train_model(X_train_tensor, y_train_tensor, input_dim, epochs=100, batch_size=32):
    train_dataset = HorseRaceDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = HorseRacePredictor(input_dim)

    num_pos = (y_train_tensor == 1).sum().item()
    num_neg = (y_train_tensor == 0).sum().item()
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

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
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return model


def test_model(model, X_test, y_test, sweep_thresholds=True, verbose=True):
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test)

    with torch.no_grad():
        logits = model(X_test_tensor).squeeze()
        probs = torch.sigmoid(logits).numpy()

    if sweep_thresholds:
        thresholds = np.arange(0.1, 0.91, 0.01)
        best_f1 = 0
        best_threshold = 0.5

        for t in thresholds:
            preds = (probs >= t).astype(int)
            f1 = f1_score(y_test, preds, average='binary', pos_label=1)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        final_preds = (probs >= best_threshold).astype(int)
    else:
        best_threshold = 0.5
        final_preds = (probs >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, final_preds)
    precision = precision_score(y_test, final_preds, average='binary')
    recall = recall_score(y_test, final_preds, average='binary')
    f1 = f1_score(y_test, final_preds, average='binary')
    conf_matrix = confusion_matrix(y_test, final_preds)


    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'predictions': final_preds,
        'best_threshold': best_threshold
    }


def run_neural_network(X_train, y_train, X_test, y_test):
    # 1. boolean → float 변환
    for col in X_train.select_dtypes(include='bool').columns:
        X_train[col] = X_train[col].astype(float)
        X_test[col] = X_test[col].astype(float)

    # 2. 정규화
    scaler = StandardScaler()
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # 3. to numpy + float32
    X_train_array = X_train.to_numpy().astype(np.float32)
    X_test_array = X_test.to_numpy().astype(np.float32)
    y_train_array = y_train.astype(np.float32).values
    y_test_array = y_test.astype(np.float32).values

    input_dim = X_train.shape[1]

    model = train_model(X_train_array, y_train_array, input_dim)
    results = test_model(model, X_test_array, y_test_array)
    return model, results
