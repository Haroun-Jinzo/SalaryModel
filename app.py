import streamlit as st
import requests
import json

# API configuration
API_URL = "http://localhost:5000/predict"

def main():
    st.title("Salary Prediction Dashboard")
    st.markdown("Predict salaries based on employee characteristics")
    
    with st.form("prediction_form"):
        st.header("Employee Information")
        
        # Input fields
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=70, value=30)
            experience = st.number_input("Years of Experience", 
                                       min_value=0, max_value=40, value=5)
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"])
            education = st.selectbox("Education Level", 
                                   ["High School", "Bachelor's", "Master's", "PhD"])
        
        job_title = st.text_input("Job Title", "Software Engineer")
        model_type = st.selectbox("Select Prediction Model",
                                options=["linear_regression", "random_forest",
                                         "xgboost", "svr", "lasso"],
                                format_func=lambda x: x.replace("_", " ").title())
        
        submitted = st.form_submit_button("Predict Salary")
        
        if submitted:
            payload = {
                "Age": age,
                "Gender": gender,
                "Education Level": education,
                "Years of Experience": experience,
                "Job Title": job_title,
                "model_type": model_type
            }
            
            try:
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    if result['status'] == 'success':
                        st.success(f"Predicted Salary: ${result['prediction']:,.2f}")
                        st.markdown(f"""
                            **Model Used**: {result['model_used'].replace('_', ' ').title()}  
                            **Input Features**:  
                            - Age: {age}  
                            - Experience: {experience} years  
                            - Gender: {gender}  
                            - Education: {education}  
                            - Job Title: {job_title}
                        """)
                    else:
                        st.error(f"Prediction Error: {result.get('error', 'Unknown error')}")
                else:
                    st.error(f"API Error (HTTP {response.status_code}): {response.text}")
            
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to prediction API. Make sure it's running!")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Add model information sidebar
    st.sidebar.header("Model Information")
    st.sidebar.markdown("""
        Available prediction models:
        - **Linear Regression**: Baseline linear model
        - **Random Forest**: Ensemble of decision trees
        - **XGBoost**: Optimized gradient boosting
        - **SVR**: Support Vector Regression
        - **Lasso**: Regularized linear model
    """)
    
    st.sidebar.header("API Status")
    try:
        health_check = requests.get(API_URL.replace('/predict', ''))
        if health_check.status_code == 404:  # Expected since root endpoint doesn't exist
            st.sidebar.success("API Connected")
        else:
            st.sidebar.warning("Unexpected API response")
    except:
        st.sidebar.error("API Not Connected")

if __name__ == '__main__':
    main()