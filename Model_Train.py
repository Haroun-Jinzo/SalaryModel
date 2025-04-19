# Importing Libraries
from matplotlib import gridspec
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
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_curve,
    auc,
    RocCurveDisplay,
    confusion_matrix
)
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import r2_score, explained_variance_score


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
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Plot')
    plt.xticks(rotation=15)
    plt.yticks(rotation=0)
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
def evaluate_model(model, x_test, y_test, ModelName=""):
    y_pred = model.predict(x_test)

    # Convert to classification by binning salaries
    y_test_class = pd.qcut(y_test, q=3, labels=["low", "medium", "high"])
    y_pred_class = pd.qcut(y_pred, q=3, labels=["low", "medium", "high"])

    # Print metrics
    print("="*50)
    print(" MODEL EVALUATION ".center(50, "="))
    print("="*50)
    print(f"\nRegression Metrics:")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"Explained Variance: {explained_variance_score(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):,.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
    
    print(f"\nClassification Metrics:")
    print(f"Precision: {precision_score(y_test_class, y_pred_class, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test_class, y_pred_class, average='weighted'):.4f}")

    visualize_performance(model ,x_test, y_test, y_pred, ModelName)
    visualize_metrics(y_test, y_pred, y_test_class, y_pred_class, ModelName)


def visualize_performance(model, X_test, y_test, y_pred, ModelName =""):
    plt.figure(figsize=(20, 10))
    
    # Calculate necessary metrics
    y_test_class = pd.qcut(y_test, q=3, labels=["low", "medium", "high"])
    y_pred_class = pd.qcut(y_pred, q=3, labels=["low", "medium", "high"])
    
    # ROC Curve calculations
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test_class)
    y_pred_bin = lb.transform(y_pred_class)
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_pred_bin.ravel())
    roc_auc = auc(fpr, tpr)
    
    # Create main title
    plt.suptitle(str(ModelName) + ' Performance Dashboard', y=0.99, fontsize=16, fontweight='bold')
    
    # Create subplots grid
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 0.7, 1])
    
    # 1. Actual vs Predicted Plot (using X_test data)
    ax1 = plt.subplot(gs[0, 0])
    sns.scatterplot(x=y_test, y=y_pred, 
                   hue=X_test['Years of Experience'],  # Using X_test data
                   palette='viridis', 
                   alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Salary')
    plt.ylabel('Predicted Salary')
    plt.title('Actual vs Predicted Values (Colored by Experience)')
    
    # Add regression metrics
    textstr = '\n'.join((
        f'R²: {r2_score(y_test, y_pred):.2f}',
        f'Explained Variance: {explained_variance_score(y_test, y_pred):.2f}',
        f'MAE: ${mean_absolute_error(y_test, y_pred):,.0f}',
        f'RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred)):,.0f}'
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, 
            verticalalignment='top', bbox=props)
    
    # 2. ROC Curve
    ax2 = plt.subplot(gs[0, 1])
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot(ax=ax2)
    plt.title(f'ROC Curve (AUC = {roc_auc:.2f})')
    
    # 3. Classification Metrics
    ax3 = plt.subplot(gs[0, 2])
    metrics_df = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'AUC'],
        'Value': [
            precision_score(y_test_class, y_pred_class, average='weighted'),
            recall_score(y_test_class, y_pred_class, average='weighted'),
            roc_auc
        ]
    })
    sns.barplot(x='Metric', y='Value', data=metrics_df, palette='rocket')
    plt.ylim(0, 1)
    plt.title('Classification Metrics')
    
    # Add value labels
    for p in ax3.patches:
        ax3.annotate(f"{p.get_height():.2f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), 
                    textcoords='offset points')
    
    # 4. Feature Importance (using model parameter)
    ax4 = plt.subplot(gs[0, 3])
    try:
        importances = model.feature_importances_
        feature_names = X_test.columns
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values('Importance', ascending=False).head(10)
        
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='Blues_d')
        plt.title('Top 10 Feature Importances')
    except AttributeError:
        plt.text(0.5, 0.5, 'Feature Importance not available\nfor this model type', 
                ha='center', va='center')
        plt.axis('off')
    
    # 5. Error Distribution
    ax5 = plt.subplot(gs[1, :])
    residuals = y_test - y_pred
    sns.histplot(residuals, kde=True, bins=30, color='#2ecc71')
    plt.axvline(0, color='#e74c3c', linestyle='--')
    plt.xlabel('Prediction Error (Actual - Predicted)')
    plt.title('Error Distribution')
    
    plt.tight_layout()
    plt.show()

from matplotlib.colors import LinearSegmentedColormap

def visualize_metrics(y_test, y_pred, y_test_class=None, y_pred_class=None, ModelName=""):
    # Calculate metrics
    metrics = {
        'Regression': {
            'MSE': mean_squared_error(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R²': r2_score(y_test, y_pred),
            'Explained Variance': explained_variance_score(y_test, y_pred)
        }
    }
    
    if y_test_class is not None and y_pred_class is not None:
        metrics['Classification'] = {
            'Precision': precision_score(y_test_class, y_pred_class, average='weighted'),
            'Recall': recall_score(y_test_class, y_pred_class, average='weighted')
        }

    # Create visualization
    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.axis('off')
    
    # Create color gradient
    cmap = LinearSegmentedColormap.from_list('metrics', ['#2ecc71', '#3498db'])
    
    # Build table data
    cell_text = []
    for category, cat_metrics in metrics.items():
        for name, value in cat_metrics.items():
            formatted_value = f"{value:.4f}" if isinstance(value, float) else f"{value:,.2f}"
            cell_text.append([category, name, formatted_value])
    
    # Create table
    table = plt.table(cellText=cell_text,
                     colLabels=['Category', 'Metric', 'Value'],
                     loc='center',
                     cellLoc='center',
                     colColours=['#f8f9fa']*3,
                     cellColours=[cmap([0.2]*3)] * len(cell_text))
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    # Highlight headers
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_facecolor('#34495e')
            cell.set_text_props(color='white', weight='bold')
    
    plt.title(str(ModelName) + ' Values Table Metrics', pad=20, fontsize=14, weight='bold')
    plt.show()

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
    #plot_distributions(df)
    
    # Detect and show outliers
    outliers = detect_outliers(df)
    print("\nDetected Salary Outliers:")
    print(outliers[['Age', 'Gender', 'Education Level', 'Job Title', 
                   'Years of Experience', 'Salary']])
    
    # Encode data properly
    encoded_df = encode_features(df.copy())
    
    # Plot correlations after encoding
    #plot_correlations(encoded_df)
    
    # Continue with training and evaluation
    lr, rfr, dtr, xgb_model,svr_model , lasso_model, x_test, y_test, train_columns, frequent_titles = train_models()
    
    # Evaluate models
    print("\nLinear Regression Evaluation:")
    evaluate_model(lr, x_test, y_test, "Linear Regression")
    
    print("\nRandom Forest Evaluation:")
    evaluate_model(rfr, x_test, y_test, "Random Forest")
    
    print("\nDecision Tree Evaluation:")
    evaluate_model(dtr, x_test, y_test, "Decision Tree")

    print("\nXGBoost evaluation")
    evaluate_model(xgb_model, x_test, y_test, "XGBoost evaluation")
    
    
    #here we print the SVR w Lasso regression algorithms
    print("\nSVR evaluation")
    evaluate_model(svr_model, x_test, y_test, "SVR evaluation")

    print("\nLasso evaluation")
    evaluate_model(lasso_model, x_test, y_test, "Lasso evaluation")