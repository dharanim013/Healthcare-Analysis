# Healthcare-Analysis
📊 Healthcare Data Analysis & Machine Learning Pipeline

A complete end-to-end data science project built using Python, designed to analyze a healthcare dataset and perform predictive modeling, anomaly detection, and AI-driven recommendation generation.

🚀 Project Overview

This project performs a full pipeline on the Healthcare Dataset (Kaggle) including:

✔️ Task 1 — Exploratory Data Analysis (EDA)

Distribution analysis using boxplots, kde plots, violin plots, and log-histograms

Frequency visualizations for:

Medical Conditions

Admission Types

Medications

Automated summary file saved under outputs/

✔️ Task 2 — Supervised Machine Learning

Target: Predict “Test Results”
Techniques used:

Feature engineering: stay duration, medication count, billing buckets

Handling high-cardinality fields (Doctor, Hospital grouping)

CatBoostClassifier (primary model)

RandomForest + OneHotEncoder fallback
Outputs:

Accuracy, precision, recall, F1

Actual vs Predicted comparison plots

Prediction CSVs

Feature importance file

Saved CatBoost model

✔️ Task 3 — Unsupervised Learning (Anomaly Detection)

Detects unusual Billing Amount values using:

Z-score analysis

IsolationForest
Generates:

Marked dataset with anomaly flags

Top anomalies CSV

Interpretation text explaining high/low billing anomalies

✔️ Task 4 — AI-Generated Doctor Recommendation (LLM-style)

Based on model predictions + patient attributes (Age, Condition, Medication):

Generates short, doctor-style recommendations

Includes actionable follow-up advice

Saves output as a text file
