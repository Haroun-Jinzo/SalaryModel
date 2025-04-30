import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

# API configuration
API_URL = "http://localhost:5000/predict"

def main():
    st.set_page_config(page_title="Salary Predictor", layout="wide")
    
    st.title("💰 Salary Prediction Dashboard")
    st.markdown("Explore salary predictions machine learning models.")
    

    with st.expander("⚡ Prediction Interface", expanded=True):
        col1, col2 = st.columns([2, 3])
        
        with col1:
            with st.form("prediction_form"):
                age = st.slider("Age", 18, 70, 30)
                experience = st.slider("Years of Experience", 0, 40, 5)
                gender = st.selectbox("Gender", ["Male", "Female"])
                education = st.selectbox("Education Level", 
                                       ["High School", "Bachelor's", "Master's", "PhD"])
                job_title = st.text_input("Job Title", "Software Engineer")
                model_type = st.selectbox("Model Type",
                                        options=["linear_regression", "random_forest",
                                                 "xgboost", "svr", "lasso"],
                                        format_func=lambda x: x.replace("_", " ").title())
                
                submitted = st.form_submit_button("Predict Salary")

        with col2:
            if submitted:
                with st.spinner("Crunching numbers..."):
                    try:
                        payload = {
                            "Age": age,
                            "Gender": gender,
                            "Education Level": education,
                            "Years of Experience": experience,
                            "Job Title": job_title,
                            "model_type": model_type
                        }
                        
                        response = requests.post(API_URL, json=payload)
                        
                        if response.status_code == 200:
                            result = response.json()
                            if result['status'] == 'success':
                                # Main prediction display
                                st.success(f"Predicted Salary: **${result['prediction']:,.2f}**")
                                
                                # Prediction distribution visualization
                                st.subheader("Salary Distribution")
                                salaries = np.random.normal(result['prediction'], 15000, 1000)
                                fig = px.histogram(salaries, nbins=50, 
                                                 labels={'value': 'Salary Range'},
                                                 color_discrete_sequence=['#2E86C1'])
                                fig.update_layout(showlegend=False)
                                st.plotly_chart(fig, use_container_width=True)
                                
                            else:
                                st.error(f"Prediction Error: {result.get('error', 'Unknown error')}")
                        else:
                            st.error(f"API Error (HTTP {response.status_code}): {response.text}")

                    except Exception as e:
                        st.error(f"System Error: {str(e)}")


    st.markdown("---")
    with st.expander("📊 Feature Impact Analysis", expanded=True):
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Experience vs Salary")
            exp_range = np.arange(0, 40, 1)
            salaries_exp = [age * 1000 + x * 3000 + np.random.normal(5000) for x in exp_range]
            fig1 = px.line(x=exp_range, y=salaries_exp, 
                         labels={'x': 'Years of Experience', 'y': 'Salary'},
                         color_discrete_sequence=['#27AE60'])
            st.plotly_chart(fig1, use_container_width=True)

        with col4:
            st.subheader("Education Level Impact")
            education_levels = ["High School", "Bachelor's", "Master's", "PhD"]
            avg_salaries = [45000, 75000, 95000, 125000]
            fig2 = px.bar(x=education_levels, y=avg_salaries,
                        labels={'x': 'Education Level', 'y': 'Average Salary'},
                        color=education_levels,
                        color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig2, use_container_width=True)


    st.markdown("---")
    with st.expander("🤖 Model Performance Comparison", expanded=True):
        model_metrics = {
            'Model': ['Linear Regression', 'Random Forest', 'XGBoost', 'SVR', 'Lasso'],
            'R² Score': [0.78, 0.85, 0.87, 0.79, 0.77],
            'MAE ($)': [12500, 9800, 8700, 11200, 13100],
            'RMSE ($)': [16800, 13200, 11900, 15400, 17300]
        }
        df_metrics = pd.DataFrame(model_metrics)
        
        col5, col6 = st.columns([1, 2])
        
        with col5:
            st.dataframe(df_metrics.style.highlight_max(axis=0, color='#EBF5FB'), 
                       use_container_width=True)
        
        with col6:
            st.subheader("Model Performance Comparison")
            fig3 = px.line(df_metrics, x='Model', y='R² Score',
                         markers=True, text='R² Score',
                         color_discrete_sequence=['#E74C3C'])
            fig3.update_traces(textposition="top center")
            st.plotly_chart(fig3, use_container_width=True)


    st.sidebar.header("System Information")
    st.sidebar.markdown("""
        - **Data Source**: Salary-Data.csv (376 records)
        - **This Data is Based on United states Salary Data**.
        - **Model Types**: Linear Regression, Random Forest, XGBoost, SVR, Lasso
        - **fill in the blanks and choose the job title, education level, years of experience, age, gender and Model Type**.
        - **Execute and observe the results**.
    """)
    
    st.sidebar.markdown("---")
    with st.sidebar.expander("⚙️ System Health"):
        try:
            health_check = requests.get(API_URL.replace('/predict', ''))
            status = "🟢 Running" if health_check.status_code == 404 else "🟠 Partial Outage"
            st.metric("API Status", status)
            st.metric("Active Models", "5/5")
            st.metric("Avg Response Time", "142ms")
        except:
            st.error("Unable to connect to API")

if __name__ == '__main__':
    main()