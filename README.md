
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
````

## Technologies

* Python
* XGBoost
* Scikit-learn
* Optuna
* River
* Streamlit
* Pandas
* NumPy

## Project Structure

```text
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
```

## Machine Learning

### Offline Prediction

The main prediction model uses **XGBoost** to classify whether a keyword is likely to become viral based on its early activity.

### Online Learning

The system also incorporates **River Adaptive Random Forest** to support continuous learning and detect changes in data patterns over time.

### Optimization

**Optuna** is used to optimize model parameters and experiment with feature configurations.

## Project Context

Team project (4 members) developed as part of the AI & Data Science engineering curriculum, with a focus on machine learning, predictive modeling, and online learning.

## Highlights

* Early viral-trend detection
* 12-hour virality prediction
* XGBoost classification
* Online learning with River
* Concept-drift detection
* Feature engineering and normalization
* Hyperparameter optimization with Optuna
* Social-media data connectors
* Streamlit-based application

## Author

**Achraf Abderrazik**

Software Engineering Student | AI & Data Science | Machine Learning


Then we'll do the **About description + topics**, and move directly to the next project.
```
