import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split


def train_model(X_train, y_train, params):
    model = KNeighborsClassifier(**params)
    model.fit(X_train, y_train)
    return model


def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'predictions': y_pred
    }

    return results


def tune_hyperparameters(X_train, y_train, X_test, y_test):
    param_grid = [
        {'n_neighbors': k, 'weights': w, 'p': p}
        for k in [3, 5, 7, 9]
        for w in ['uniform', 'distance']
        for p in [1, 2]
    ]

    best_f1 = -np.inf
    best_params = None
    best_results = None

    for params in param_grid:
        model = train_model(X_train, y_train, params)
        results = test_model(model, X_test, y_test)

        if results['f1_score'] > best_f1:
            best_f1 = results['f1_score']
            best_params = params
            best_results = results

    return best_params, best_results


def main(data_path, target_column):
    data = pd.read_csv(data_path)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    best_params, best_results = tune_hyperparameters(X_train, y_train, X_test, y_test)

    print("Best Hyperparameters:", best_params)
    print("Best Results:", best_results)


if __name__ == "__main__":
    main('data/processed_race_results.csv', 'target')
