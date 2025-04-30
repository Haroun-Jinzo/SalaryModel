from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load preprocessing metadata
with open('preprocessing_metadata.pkl', 'rb') as f:
    preprocessing = pickle.load(f)

# Load all models
models = {
    'linear_regression': pickle.load(open('Linear_Regression.pkl', 'rb')),
    'random_forest': pickle.load(open('Random_Forest.pkl', 'rb')),
    'xgboost': pickle.load(open('XGBoost.pkl', 'rb')),
    'svr': pickle.load(open('SVR.pkl', 'rb')),
    'lasso': pickle.load(open('Lasso.pkl', 'rb'))
}

def preprocess_input(data):
    input_df = pd.DataFrame([{
        'Age': data['Age'],
        'Years of Experience': data['Years of Experience'],
        'Gender': data['Gender'],
        'Education Level': data['Education Level'],
        'Job Title': data['Job Title']
    }])

    # 1. Process Education Level
    input_df['Education Level'] = input_df['Education Level'].map(
        preprocessing['education_mapping']
    ).fillna(0)  # Handle unknown education levels
    
    # 2. Encode Gender
    try:
        input_df['Gender'] = preprocessing['label_encoder'].transform(
            input_df['Gender'].str.strip().str.lower()
        )
    except ValueError:
        input_df['Gender'] = 0  # Fallback to first category
    
    # 3. Process Job Title
    input_df['Job Title'] = input_df['Job Title'].apply(
        lambda x: x if x in preprocessing['frequent_titles'] else 'Others'
    )

    # 4. Create aligned dummy variables
    dummy_template = pd.DataFrame(columns=preprocessing['job_title_columns'])
    current_dummies = pd.get_dummies(input_df[['Job Title']], prefix='Job Title')
    aligned_dummies = dummy_template.combine_first(current_dummies).fillna(0)

    # 5. Build final feature array
    final_features = pd.concat([
        input_df[['Age', 'Years of Experience', 'Gender', 'Education Level']],
        aligned_dummies
    ], axis=1)

    # 6. Ensure exact column order from training
    return final_features[preprocessing['feature_columns']]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Validate input
        required_fields = ['Age', 'Gender', 'Education Level', 
                          'Years of Experience', 'Job Title']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Get model type
        model_type = data.get('model_type', 'linear_regression')
        if model_type not in models:
            return jsonify({'error': 'Invalid model type'}), 400
        
        # Preprocess input
        processed_data = preprocess_input(data)
        
        # Validate feature dimensions
        if processed_data.shape[1] != len(preprocessing['feature_columns']):
            return jsonify({'error': 'Feature dimension mismatch'}), 400
        
        # Make prediction
        prediction = models[model_type].predict(processed_data)[0]
        
        return jsonify({
            'prediction': round(float(prediction), 2),
            'model_used': model_type,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)