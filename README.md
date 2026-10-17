
# Customer Churn Analytics & Prediction

An interactive Machine Learning web application that analyzes customer churn patterns and predicts whether a telecom customer is likely to leave the service.

## 🚀 Live Demo

**Try the application:** https://churn-analytics-dashboard1.streamlit.app/

## 📌 Project Overview

Customer churn is a major challenge for telecommunications companies. Understanding why customers leave helps businesses identify at-risk customers and make better retention decisions.

This project combines exploratory data analysis, machine learning, and an interactive Streamlit dashboard to analyze customer behavior and predict churn.

## 🎯 Objectives

- Analyze customer churn patterns.
- Explore the relationship between customer characteristics, services, contracts, and churn.
- Build a machine learning pipeline to predict customer churn.
- Present insights through an interactive dashboard.
- Allow users to enter customer details and obtain churn predictions.

## 📊 Dataset

- **Dataset:** IBM Telco Customer Churn
- **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Target variable:** `Churn`
- **Classes:** Yes (customer churned) and No (customer retained)

The dataset contains customer demographic information, subscribed services, contract details, billing information, and churn status.

## 🛠️ Tech Stack

- **Python** — Core programming language
- **Pandas** — Data manipulation
- **Scikit-learn** — Preprocessing and machine learning
- **Imbalanced-learn** — SMOTE for handling class imbalance
- **Joblib** — Saving and loading the trained model
- **Streamlit** — Interactive web application
- **Matplotlib / Seaborn** — Data visualization, where used in the analysis

## 🤖 Machine Learning Workflow

1. Load and inspect the customer dataset.
2. Prepare the features and target variable.
3. Preprocess numerical and categorical features using a ColumnTransformer.
4. Apply SMOTE to address class imbalance during model training.
5. Train a Logistic Regression classifier.
6. Save the trained pipeline as `churn_model.pkl`.
7. Load the saved model in the Streamlit application.
8. Generate churn predictions and display the results.

## ✨ Features

### Interactive Dashboard
- Total customer count
- Churned and retained customer counts
- Overall churn rate
- Contract-based churn analysis
- Monthly charge comparisons
- Dataset preview and data-quality information

### Churn Prediction
- Customer profile inputs
- Service and subscription selections
- Contract and billing information
- Churn prediction with estimated churn probability
- Basic risk guidance based on the prediction

## 📂 Project Structure

```text
churn-analytics/
├── app.py
├── churn_model.pkl
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── requirements.txt
└── README.md
```
## 👨‍💻 Project Focus

This project demonstrates practical skills in data analysis, feature preprocessing, handling imbalanced data, machine learning, model serialization, and web application deployment.
