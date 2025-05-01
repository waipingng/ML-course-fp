"""
logistic.py - Implementation of Logistic Regression

This module provides functions to train and test a logistic regression model
on given datasets.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix


def train_model(X_train, y_train):
    # Initialize the model with suitable parameters
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model


def test_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Return results as a dictionary
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'predictions': y_pred
    }
    
    return results


def run_logistic_regression(X_train, y_train, X_test, y_test):
    # Train model
    model = train_model(X_train, y_train)
    
    # Test model
    results = test_model(model, X_test, y_test)
    
    return model, results
