# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import pickle


# here we load data put csv file next to this script
def load_data():
    df = pd.read_csv('Salary-Data.csv')
    return df

#here we process the data we remove the title columns
def preprocess_data(df):
    df.dropna(inplace=True)
    
    # Process Job Titles
    job_title_count = df['Job Title'].value_counts()
    job_title_edited = job_title_count[job_title_count <= 25]
    df['Job Title'] = df['Job Title'].apply(lambda x: 'Others' if x in job_title_edited else x)
    
    # Process Education Levels
    df['Education Level'].replace(["Bachelor's Degree", "Master's Degree", "phD"],
                                 ["Bachelor's", "Master's", "PhD"], inplace=True)
    return df

def encode_features(df):
    education_mapping = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
    df['Education Level'] = df['Education Level'].map(education_mapping)
    
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    
    # Create dummy variables
    dummies = pd.get_dummies(df['Job Title'], drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    df.drop('Job Title', axis=1, inplace=True)
    return df


#here we create the visualiation data for now khalih lena im willing bch n7otou fel streamlit code maa 
def plot_distributions(df):
    # Create a copy for visualization
    viz_df = df.copy()
    
    # Temporary encoding for visualization only
    viz_df['Gender'] = viz_df['Gender'].map({'Female': 0, 'Male': 1})
    
    # Original plotting code without correlation plot
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    sns.countplot(x='Gender', data=df, ax=ax[0])
    sns.countplot(x='Education Level', data=df, ax=ax[1])
    ax[0].set_title('Distribution of Gender')
    ax[1].set_title('Distribution of Education Level')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(3, 1, figsize=(12, 15))
    sns.histplot(df['Age'], ax=ax[0], color='blue', kde=True)
    sns.histplot(df['Years of Experience'], ax=ax[1], color='orange', kde=True)
    sns.histplot(df['Salary'], ax=ax[2], color='green', kde=True)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    sns.barplot(x='Gender', y='Salary', data=df, ax=ax[0])
    sns.boxplot(x='Education Level', y='Salary', data=df, ax=ax[1])
    ax[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    sns.regplot(x='Age', y='Salary', data=df, ax=ax[0])
    sns.regplot(x='Years of Experience', y='Salary', data=df, ax=ax[1])
    plt.tight_layout()
    plt.show()

def plot_correlations(df):
    # Now using fully encoded data
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Plot')
    plt.xticks(rotation=60)
    plt.yticks(rotation=60)
    plt.show()

# IDK what this do Im gonna ask oumeyma XD
def detect_outliers(df):
    Q1 = df.Salary.quantile(0.25)
    Q3 = df.Salary.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df.Salary > upper) | (df.Salary < lower)]

#Trainin here
def train_models():
    df = load_data()
    df = preprocess_data(df)
    
    # Save frequent titles before encoding
    frequent_titles = df['Job Title'].value_counts()[
        df['Job Title'].value_counts() > 25
    ].index.tolist()
    
    df = encode_features(df)
    
    # Save feature names
    train_columns = df.drop('Salary', axis=1).columns.tolist()
    
    features = df.drop('Salary', axis=1)
    target = df['Salary']
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=42
    )
    
    model_params = {
        'Linear_Regression': {'model': LinearRegression(), 'params': {}},
        'Decision_Tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'max_depth': [2,4,6,8,10],
                'random_state': [0,42],
                'min_samples_split': [1,5,10,20]
            }
        },
        'Random_Forest': {
            'model': RandomForestRegressor(),
            'params': {'n_estimators': [10,30,20,50,80]}
        },
        'XGBoost' : 
        {
          'model': xgb.XGBRegressor(),
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0]
            }  
        },
         #here put SVR w l lasso regression model
         'SVR': {
            'model': SVR(),
            'params': {
                'kernel': ['linear', 'rbf'],
                'C': [0.1, 1, 10],
                'epsilon': [0.1, 0.2, 0.5]
            }
        },
        'Lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [0.01, 0.1, 1.0, 10.0]
            }
        }
       
    }
    
    scores = []
    for model_name, mp in model_params.items():
        clf = GridSearchCV(mp['model'], mp['params'], cv=5, scoring='neg_mean_squared_error')
        clf.fit(x_train, y_train)
        scores.append({
            'Model': model_name,
            'Params': clf.best_params_,
            'MSE(-ve)': clf.best_score_,
        })
    
    # Initialize and train all models properly
    lr = LinearRegression()  # This was missing
    lr.fit(x_train, y_train)
    
    rfr = RandomForestRegressor(n_estimators=20)
    rfr.fit(x_train, y_train)
    
    dtr = DecisionTreeRegressor(max_depth=10, min_samples_split=2, random_state=0)
    dtr.fit(x_train, y_train)

    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    xgb_model.fit(x_train, y_train)
    
    # here we will fit the SVR w lasso regression Model n7otou l value fe return
    svr_model = SVR(kernel='linear', C=1.0, epsilon=0.1)
    svr_model.fit(x_train, y_train)

    lasso_model = Lasso(alpha=0.1)
    lasso_model.fit(x_train, y_train)

    return lr, rfr, dtr, xgb_model, svr_model , lasso_model, x_test, y_test, train_columns, frequent_titles


# this is evaluation
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))

def save_model(model):
    pickle.dump(model, open('model.pkl', 'wb'))

def process_data(entry, train_columns=None, frequent_titles=None):
    entry_df = pd.DataFrame([entry])
    
    # Education Level mapping
    education_mapping = {"High School":0, "Bachelor's":1, "Master's":2, "PhD":3}
    entry_df['Education Level'] = entry_df['Education Level'].map(education_mapping)
    
    # Gender encoding
    le = LabelEncoder()
    le.classes_ = np.array(['Female', 'Male'])
    entry_df['Gender'] = le.transform(entry_df['Gender'])
    
    # Job Title processing - use the same threshold as training
    entry_df['Job Title'] = entry_df['Job Title'].apply(
        lambda x: x if x in frequent_titles else 'Others')
    
    # Create dummies aligned with training data
    dummies = pd.get_dummies(entry_df[['Job Title']], prefix='Job Title')
    
    # Ensure we have all columns from training
    if train_columns is not None:
        missing_cols = set(train_columns) - set(dummies.columns)
        for col in missing_cols:
            dummies[col] = 0
        dummies = dummies[train_columns]
        
    entry_df = pd.concat([entry_df.drop('Job Title', axis=1), dummies], axis=1)
    return entry_df

def predict_salary(model, processed_data):
    return model.predict(processed_data)[0]

# Execute pipeline l main function win we will call the previous functions
if __name__ == '__main__':
    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)
    
    # Plot distributions with original data
    plot_distributions(df)
    
    # Detect and show outliers
    outliers = detect_outliers(df)
    print("\nDetected Salary Outliers:")
    print(outliers[['Age', 'Gender', 'Education Level', 'Job Title', 
                   'Years of Experience', 'Salary']])
    
    # Encode data properly
    encoded_df = encode_features(df.copy())
    
    # Plot correlations after encoding
    plot_correlations(encoded_df)
    
    # Continue with training and evaluation
    lr, rfr, dtr, xgb_model,svr_model , lasso_model, x_test, y_test, train_columns, frequent_titles = train_models()
    
    # Evaluate models
    print("\nLinear Regression Evaluation:")
    evaluate_model(lr, x_test, y_test)
    
    print("\nRandom Forest Evaluation:")
    evaluate_model(rfr, x_test, y_test)
    
    print("\nDecision Tree Evaluation:")
    evaluate_model(dtr, x_test, y_test)

    print("\nXGBoost evaluation")
    evaluate_model(xgb_model, x_test, y_test)
    
    
    #here we print the SVR w Lasso regression algorithms
    print("\nSVR evaluation")
    evaluate_model(svr_model, x_test, y_test)
    
    print("\nLasso evaluation")
    evaluate_model(lasso_model, x_test, y_test)