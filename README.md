# 💼 Salary Prediction ML System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Supervised-orange)
![Deployment](https://img.shields.io/badge/Deployment-Flask%20%2B%20Streamlit-green)

An end-to-end machine learning system for salary prediction using employee demographics, featuring model training, REST API, and interactive dashboard.

## 🌟 Features
- 5 Regression Models (Linear Regression, Random Forest, XGBoost, SVR, Lasso)
- Flask REST API for predictions
- Streamlit Web Dashboard
- Automated Feature Engineering
- Hyperparameter Tuning with GridSearchCV
- Model Persistence with Pickle

## 📂 Project Structure

## 🔄 Workflow Diagram

```mermaid
graph TD
    A[Raw Data] --> B{Preprocessing}
    B -->|Clean & Transform| C[Processed Data]
    C --> D{Feature Engineering}
    D -->|Encode & Create Features| E[Training Data]
    E --> F[Model Training]
    F --> G[Model Evaluation]
    G --> H[Model Deployment]
    H --> I[API Predictions]
    H --> J[Streamlit Dashboard]
