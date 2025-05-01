import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HorseRaceDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.LongTensor(targets)  # CrossEntropyLoss를 위해 LongTensor 사용

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]



class HorseRacePredictor(nn.Module):
    def __init__(self, input_dim):
        super(HorseRacePredictor, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.layer2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.layer3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)

        self.layer4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)

        self.output = nn.Linear(32, 16)

        self.dropout = nn.Dropout(0.4)
        self.act = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.dropout(self.act(self.bn1(self.layer1(x))))
        x = self.dropout(self.act(self.bn2(self.layer2(x))))
        x = self.dropout(self.act(self.bn3(self.layer3(x))))
        x = self.dropout(self.act(self.bn4(self.layer4(x))))
        return self.output(x)





def train_model(X_train_tensor, y_train_tensor, input_dim, epochs=400, batch_size=32):
    train_dataset = HorseRaceDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = HorseRacePredictor(input_dim).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # label smoothing 추가
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4, amsgrad=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return model





def test_model(model, X_test, y_test, verbose=True):
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        logits = model(X_test_tensor)  # [batch, 16]
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average='macro')  # multi-class 평가
    recall = recall_score(y_test, preds, average='macro')
    f1 = f1_score(y_test, preds, average='macro')
    conf_matrix = confusion_matrix(y_test, preds)

    if verbose:
        print(f"Accuracy: {accuracy:.4f}, F1 (macro): {f1:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'predictions': preds
    }
    
    


def run_neural_network(X_train, y_train, X_test, y_test):
    print(f"Using device: {device}")

    # 정규화: 수치형 feature만 스케일링
    scaler = StandardScaler()
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Tensor 변환
    X_train_array = X_train.to_numpy().astype(np.float32)
    X_test_array = X_test.to_numpy().astype(np.float32)
    y_train_array = y_train.astype(int).values  # CrossEntropyLoss를 위해 int
    y_test_array = y_test.astype(int).values

    input_dim = X_train.shape[1]

    # 학습
    model = train_model(X_train_array, y_train_array, input_dim)

    # 평가
    results = test_model(model, X_test_array, y_test_array)
    return model, results
