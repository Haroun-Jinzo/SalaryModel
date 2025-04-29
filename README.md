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

## Introduction
- Salary prediction is crucial for the labor market, career management, and workforce planning.

- Traditional methods often rely on surveys and historical data, which may be biased or not scalable.

- Machine learning (ML) models provide more accurate salary predictions compared to traditional methods.


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


    ## 🔄 Sequence Diagram
    ```mermaid
    sequenceDiagram
        A[User]->>B[Streamlit]: Submit Form
        B[Streamlit]->>C[Flask]: POST /predict
        C[Flask]->>D[Model]: Process Request
        D[Model]->>C[Flask]: Return Prediction
        C[Flask]->>B[Streamlit]: JSON Response
        B[Streamlit]->>A[User]: Show Result

  

### Evaluation Metrics Table
| Model              | R²     | Precision | Recall | Explained Variance |
|--------------------|--------|-----------|--------|---------------------|
| Linear Regression  | 0.8884 | 0.8508    | 0.8511 | 0.8901             |
| Random Forest      | 0.8707 | 0.8957    | 0.8936 | 0.8751             |
| Decision Tree      | 0.8594 | 0.8623    | 0.8617 | 0.8649             |
| Xgboost            | 0.8772 | 0.8856    | 0.8830 | 0.8786             |
| SVR                | 0.5513 | 0.8398    | 0.8404 | 0.5570             |
| Lasso              | 0.8884 | 0.8508    | 0.8511 | 0.8901             |

