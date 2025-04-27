# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import pickle

def load_data():
    return pd.read_csv('Salary-Data.csv')

def preprocess_data(df):
    df = df.dropna().copy()
    job_title_count = df['Job Title'].value_counts()
    df['Job Title'] = np.where(df['Job Title'].isin(job_title_count[job_title_count <= 25].index), 
                              'Others', df['Job Title'])
    df['Education Level'] = df['Education Level'].replace({
        "Bachelor's Degree": "Bachelor's",
        "Master's Degree": "Master's",
        "phD": "PhD"
    })
    return df, job_title_count[job_title_count > 25].index.tolist()

def encode_features(df):
    df = df.copy()
    education_mapping = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
    df['Education Level'] = df['Education Level'].map(education_mapping)
    
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    
    dummies = pd.get_dummies(df['Job Title'], prefix='Job Title')
    job_title_columns = dummies.columns.tolist()  # Get actual dummy columns
    
    if 'Job Title_Others' not in job_title_columns:
        dummies['Job Title_Others'] = 0
    
    df = pd.concat([df.drop('Job Title', axis=1), dummies], axis=1)
    return df, df.drop('Salary', axis=1).columns.tolist(), education_mapping, le, job_title_columns

def train_and_save_models(x_train, y_train):
    model_configs = {
        # ... keep existing model configs unchanged ...
    }
    
    for model_name, config in model_configs.items():
        grid_search = GridSearchCV(config['model'], config['params'], cv=5, 
                                 scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(x_train, y_train)
        with open(f'{model_name}.pkl', 'wb') as f:
            pickle.dump(grid_search.best_estimator_, f)

def main():
    df, frequent_titles = preprocess_data(load_data())
    encoded_df, feature_columns, education_mapping, label_encoder, job_title_columns = encode_features(df)
    
    x_train, x_test, y_train, y_test = train_test_split(
        encoded_df.drop('Salary', axis=1), encoded_df['Salary'], test_size=0.25, random_state=42
    )
    
    # Save metadata with job_title_columns
    with open('preprocessing_metadata.pkl', 'wb') as f:
        pickle.dump({
            'feature_columns': feature_columns,
            'job_title_columns': job_title_columns,
            'frequent_titles': frequent_titles,
            'education_mapping': education_mapping,
            'label_encoder': label_encoder
        }, f)
    
    train_and_save_models(x_train, y_train)
    
    # Evaluation code remains the same

if __name__ == '__main__':
    main()