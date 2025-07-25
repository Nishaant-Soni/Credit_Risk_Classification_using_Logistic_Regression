import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load all trained models and feature names
@st.cache_resource
def load_models():
    models = {}
    
    # Load individual models
    try:
        models['Logistic Regression'] = joblib.load('logistic_regression_model.pkl')
    except FileNotFoundError:
        models['Logistic Regression'] = joblib.load('credit_risk_model.pkl')  # fallback
    
    try:
        models['Random Forest'] = joblib.load('random_forest_model.pkl')
    except FileNotFoundError:
        models['Random Forest'] = models['Logistic Regression']  # fallback
    
    try:
        models['XGBoost'] = joblib.load('xgboost_model.pkl')
    except FileNotFoundError:
        models['XGBoost'] = models['Logistic Regression']  # fallback
    
    # Load ensemble models if available
    try:
        models['Soft Voting (LR+RF+XGB)'] = joblib.load('ensemble_soft_voting_lr_rf_xgb.pkl')
    except FileNotFoundError:
        pass
    
    # Load model info
    try:
        model_info = joblib.load('model_info.pkl')
    except FileNotFoundError:
        model_info = None
    
    feature_names = joblib.load('feature_names.pkl')
    
    return models, model_info, feature_names

# Load models
models, model_info, feature_names = load_models()

# App title and description
st.title("🏦 Credit Risk Assessment Tool")
st.markdown("""
This application uses multiple machine learning models to assess credit risk for loan applications.
Choose from **Logistic Regression**, **Random Forest**, **XGBoost**, and **Ensemble models** trained on 
historical lending data with advanced techniques like oversampling to handle class imbalance.

**Random Forest** is set as the default model as it achieved the highest performance in our testing.
""")

# Create sidebar for input
st.sidebar.header("Model Selection")

# Model selector
available_models = list(models.keys())
default_model = 'Random Forest' if 'Random Forest' in available_models else available_models[0]

selected_model_name = st.sidebar.selectbox(
    "Choose Model",
    available_models,
    index=available_models.index(default_model) if default_model in available_models else 0,
    help="Select which machine learning model to use for prediction"
)

selected_model = models[selected_model_name]

# Display model performance if available
if model_info and 'model_comparison' in model_info:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model Performance:**")
    
    # Find performance metrics for selected model
    for model_data in model_info['model_comparison']:
        if model_data['Model'] == selected_model_name:
            st.sidebar.metric("Accuracy", f"{model_data['Test Accuracy']:.3f}")
            st.sidebar.metric("ROC AUC", f"{model_data['ROC AUC']:.3f}")
            break

st.sidebar.header("Loan Application Details")

# Input fields
loan_size = st.sidebar.number_input(
    "Loan Size ($)", 
    min_value=0.0, 
    max_value=100000.0, 
    value=10000.0,
    step=500.0,
    help="The total amount of the loan being requested"
)

interest_rate = st.sidebar.number_input(
    "Interest Rate (%)", 
    min_value=0.0, 
    max_value=25.0, 
    value=7.5,
    step=0.1,
    help="The interest rate for the loan"
)

borrower_income = st.sidebar.number_input(
    "Borrower Income ($)", 
    min_value=0.0, 
    max_value=200000.0, 
    value=50000.0,
    step=1000.0,
    help="Annual income of the borrower"
)

debt_to_income = st.sidebar.number_input(
    "Debt-to-Income Ratio", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.4,
    step=0.01,
    help="Ratio of total debt to income"
)

num_of_accounts = st.sidebar.number_input(
    "Number of Accounts", 
    min_value=0, 
    max_value=20, 
    value=5,
    step=1,
    help="Total number of credit accounts"
)

derogatory_marks = st.sidebar.number_input(
    "Derogatory Marks", 
    min_value=0, 
    max_value=10, 
    value=0,
    step=1,
    help="Number of derogatory marks on credit report"
)

total_debt = st.sidebar.number_input(
    "Total Debt ($)", 
    min_value=0.0, 
    max_value=100000.0, 
    value=20000.0,
    step=500.0,
    help="Total debt amount"
)

# Create prediction button
if st.sidebar.button("Assess Credit Risk", type="primary"):
    # Prepare input data
    input_data = pd.DataFrame({
        'loan_size': [loan_size],
        'interest_rate': [interest_rate],
        'borrower_income': [borrower_income],
        'debt_to_income': [debt_to_income],
        'num_of_accounts': [num_of_accounts],
        'derogatory_marks': [derogatory_marks],
        'total_debt': [total_debt]
    })
    
    # Make prediction
    prediction = selected_model.predict(input_data)[0]
    prediction_proba = selected_model.predict_proba(input_data)[0]
    
    # Display results
    st.header(f"Risk Assessment Results - {selected_model_name}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if prediction == 0:
            st.success("✅ **HEALTHY LOAN**")
            st.markdown("This loan application is classified as **low risk**.")
        else:
            st.error("⚠️ **HIGH-RISK LOAN**")
            st.markdown("This loan application is classified as **high risk**.")
    
    with col2:
        st.metric(
            "Confidence Score", 
            f"{max(prediction_proba):.1%}",
            help="Model confidence in the prediction"
        )
    
    with col3:
        st.metric(
            "Model Type",
            selected_model_name,
            help="The machine learning model used for this prediction"
        )
    
    # Show probability breakdown
    st.subheader("Risk Probability Breakdown")
    prob_df = pd.DataFrame({
        'Risk Level': ['Healthy Loan', 'High-Risk Loan'],
        'Probability': [prediction_proba[0], prediction_proba[1]]
    })
    
    st.bar_chart(prob_df.set_index('Risk Level'))
    
    # Show input summary
    st.subheader("Application Summary")
    summary_df = pd.DataFrame({
        'Feature': [
            'Loan Size', 'Interest Rate', 'Borrower Income', 
            'Debt-to-Income Ratio', 'Number of Accounts', 
            'Derogatory Marks', 'Total Debt'
        ],
        'Value': [
            f"${loan_size:,.0f}", 
            f"{interest_rate}%", 
            f"${borrower_income:,.0f}",
            f"{debt_to_income:.2f}", 
            f"{num_of_accounts}", 
            f"{derogatory_marks}", 
            f"${total_debt:,.0f}"
        ]
    })
    st.table(summary_df)

# Model comparison section
if model_info and 'model_comparison' in model_info:
    with st.expander("📊 Model Performance Comparison"):
        st.markdown("**Compare all trained models:**")
        
        comparison_df = pd.DataFrame(model_info['model_comparison'])
        
        # Format the dataframe for better display
        comparison_display = comparison_df.copy()
        comparison_display['Test Accuracy'] = comparison_display['Test Accuracy'].apply(lambda x: f"{x:.3f}")
        comparison_display['ROC AUC'] = comparison_display['ROC AUC'].apply(lambda x: f"{x:.3f}")
        comparison_display['CV Mean'] = comparison_display['CV Mean'].apply(lambda x: f"{x:.3f}")
        
        st.dataframe(comparison_display, use_container_width=True)
        
        # Highlight best performers
        best_accuracy = comparison_df['Test Accuracy'].max()
        best_roc_auc = comparison_df['ROC AUC'].max()
        
        st.markdown(f"""
        **Best Performers:**
        - **Highest Accuracy**: {comparison_df.loc[comparison_df['Test Accuracy'].idxmax(), 'Model']} ({best_accuracy:.3f})
        - **Highest ROC AUC**: {comparison_df.loc[comparison_df['ROC AUC'].idxmax(), 'Model']} ({best_roc_auc:.3f})
        """)

# Model information
with st.expander("ℹ️ About the Models"):
    st.markdown(f"""
    **Available Models:**
    - **Logistic Regression**: Linear model with regularization
    - **Random Forest**: Ensemble of decision trees (recommended)
    - **XGBoost**: Gradient boosting algorithm
    - **Ensemble Models**: Soft Voting classifier combining multiple models
    
    **Currently Selected**: {selected_model_name}
    
    **Training Details:**
    - Data preprocessing with oversampling for class balance
    - 5-fold cross-validation for robust evaluation
    - Features: 7 financial indicators
    - Performance metrics: Accuracy, ROC AUC, Cross-validation scores
    
    **Features Used:**
    - Loan Size: Amount of loan requested
    - Interest Rate: Loan interest rate
    - Borrower Income: Annual income
    - Debt-to-Income Ratio: Existing debt relative to income
    - Number of Accounts: Total credit accounts
    - Derogatory Marks: Negative credit history items
    - Total Debt: Current total debt amount
    
    **Model Selection Guidance:**
    - **Random Forest**: Best overall performance, handles non-linear relationships
    - **XGBoost**: Strong performance, good for complex patterns
    - **Logistic Regression**: Interpretable, good baseline
    - **Ensemble**: Combines Random Forest, XGBoost and Logistic Regression models for robust predictions

    **Disclaimer:** This tool is for educational purposes only and should not be used for actual lending decisions.
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Multi-Model Credit Risk Assessment | Logistic Regression • Random Forest • XGBoost • Ensemble Methods")
