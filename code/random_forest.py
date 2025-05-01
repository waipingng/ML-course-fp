"""
random_forest.py - Implementation of Random Forest Classifier

This module provides functions to train and test a random forest model
on given datasets, with optional hyperparameter tuning.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=None, 
        random_state=42, 
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model


def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
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


def tune_model(X_train, y_train, cv=3):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight='balanced'),
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    print("Best parameters found:", grid_search.best_params_)
    return grid_search.best_estimator_


def run_random_forest(X_train, y_train, X_test, y_test, tune=False):
    if tune:
        model = tune_model(X_train, y_train)
    else:
        model = train_model(X_train, y_train)
    
    results = test_model(model, X_test, y_test)
    return model, results
