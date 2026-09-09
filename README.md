# TrendRadar — Early Detection of Viral Trends

> Machine learning system for predicting a keyword's future virality from its first 15 minutes of activity.

## Overview

TrendRadar is a machine learning project designed to detect emerging viral trends at an early stage.

The system analyzes the initial activity of a keyword and predicts its **12-hour virality**, helping identify trends before they become widely popular.

## Key Results

- Trained an **XGBoost** model on approximately **179,000 tweets**.
- Achieved a **ROC-AUC of 0.921** for early trend prediction.
- Applied **Optuna** for hyperparameter optimization and feature experimentation.
- Extended the system with **River Adaptive Random Forest** for online learning and concept-drift detection.

## Architecture

```text
Social Media Activity
        ↓
Feature Extraction
        ↓
Normalization
        ↓
XGBoost Trend Prediction
        ↓
12-Hour Virality Prediction
        ↓
Online Learning / Concept Drift Detection
        ↓
TrendRadar

Technologies
Python
XGBoost
Scikit-learn
Optuna
River
Streamlit
Pandas
NumPy
Project Structure
TrendRadar/
├── app.py
├── extractor.py
├── model_manager.py
├── normalizer.py
├── feedback_queue.py
├── warm_up_model.py
├── test_key.py
├── requirements.txt
├── connectors/
│   ├── base.py
│   ├── bluesky_connector.py
│   └── mastodon_connector.py
├── trendradar_xgb_model.pkl
├── trendradar_river_model.pkl
└── trendradar_normalizer.pkl
Machine Learning
Offline Prediction

The main prediction model uses XGBoost to classify whether a keyword is likely to become viral based on its early activity.

Online Learning

The system also incorporates River Adaptive Random Forest to support continuous learning and detect changes in data patterns over time.

Optimization

Optuna is used to optimize model parameters and experiment with feature configurations.

Project Context

This project was developed as part of a Master's / engineering AI & Data Science curriculum, with a focus on machine learning, predictive modeling, and online learning.

Author

Achraf Abderrazik

Software Engineering Student | AI & Data Science | Machine Learning


### What you do now

On your **TrendRadar GitHub page**, click:

**Add a README → paste the text above → Commit changes**

Don't upload anything else yet. Your `.pkl` files are already in the repository, as shown in your screenshot.

After you add the README, send me the screenshot of the repository page and we'll move to the **next repo
