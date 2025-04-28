# 💼 Salary Prediction Machine Learning System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Supervised-orange)
![Deployment](https://img.shields.io/badge/Deployment-Flask%20%2B%20Streamlit-green)

An end-to-end system for predicting salaries based on employee demographics using multiple regression models.

## 📊 Dataset Overview
**Salary-Data.csv** (376 records) contains:
- **Features**: Age, Gender, Education Level, Job Title, Years of Experience
- **Target**: Salary (USD)
- **Preprocessing**:
  - Missing values removed
  - Rare job titles (<25 occurrences) grouped as "Others"
  - Education levels standardized
  - Categorical features encoded

# 💼 Salary Prediction Machine Learning System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Supervised-orange)
![Deployment](https://img.shields.io/badge/Deployment-Flask%20%2B%20Streamlit-green)

An end-to-end system for predicting salaries based on employee demographics using multiple regression models.

## 📊 Dataset Overview
**Salary-Data.csv** (376 records) contains:
- **Features**: Age, Gender, Education Level, Job Title, Years of Experience
- **Target**: Salary (USD)
- **Preprocessing**:
  - Missing values removed
  - Rare job titles (<25 occurrences) grouped as "Others"
  - Education levels standardized
  - Categorical features encoded

## 🔄 Workflow Diagram
  ```mermaid
  graph TD
      A[Raw Data] --> B{Preprocessing}
      B -->|Clean Data| C[Feature Engineering]
      C -->|Encode Features| D[Train-Test Split]
      D -->|75% Training| E[Model Training]
      D -->|25% Testing| F[Model Evaluation]
      E --> G[Save Models]
      F --> H[Performance Metrics]

  ```mermaid
  sequenceDiagram
      A[User]->>B[Streamlit]: Submit Form
      B[Streamlit]->>C[Flask]: POST /predict
      C[Flask]->>D[Model]: Process Request
      D[Model]->>C[Flask]: Return Prediction
      C[Flask]->>B[Streamlit]: JSON Response
      B[Streamlit]->>A[User]: Show Result
