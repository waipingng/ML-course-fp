<p align="center">
  <img src="figure/horse_racing.webp" alt="ML-course-fp" width="600"/>
</p>


# 🐎 Horse Racing Prediction Project

## Introduction

In this project, we aim to build a classification model to predict horse racing outcomes using historical data from the Japan Jockey Association (JRA). Our primary goal is to determine whether a horse will finish in the top positions of a race and to evaluate different machine learning algorithms for predictive performance and interpretability.

## Problem Statement

We formulate this as a multiclass classification problem:

    Top 1: Horse wins the race

    Top 3: Horse places among the top three

    Not Top 3: Horse finishes outside the top three

The broader goals include:

    Identifying the most predictive features of race outcomes.

    Evaluating and comparing multiple machine learning models.

    Recommending the best model based on a balance of accuracy and interpretability.

## Data Overview

    Source: Japan Jockey Club (JRA)

    Collection: Web-scraped historical data

    Granularity: Each row = one horse’s result in one race

## Features

-Race ID | Unique identifier for the race
-Race Type | Distance of race in meters
-Weather | Encoded weather condition (e.g., Weather01)
-Grade | Race classification (e.g., G1, G2, G3)
-Finish Position | Horse’s rank in the race
-Horse ID | Unique identifier for each horse
-Age/Sex | Horse's age and sex (e.g., 3F = 3-year-old filly)
-Final Time | Time to finish the race
-Odds | Betting odds prior to the race
-Horse Weight (kg) | Weight of the horse (excluding jockey)

## Feature Engineering

-Transformed Final Time from mm:ss.s to total seconds.

- Parsed Age/Sex into two features: numeric age and categorical sex.

- One-hot encoded Grade, Weather, and Race Type.

- Created binary target variables for Top 1 and Top 3 finishes.

- Normalized Odds and Horse Weight.

## Reproduction

## Modeling & Evaluation

We tested the following models:

    Logistic Regression (baseline)

    Decision Tree

    K-Nearest Neighbors

    Random Forest

    XGBoost (advanced, newly introduced technique)

    Neural Network (PyTorch)


## Metrics

Models were evaluated using:

    Accuracy

    ROC-AUC Score

    Average Precision

    Confusion Matrix

## Results

## Limitations

## Conclusion/Recommendation

## Appendix
