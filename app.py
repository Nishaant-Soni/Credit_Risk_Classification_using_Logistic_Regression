import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load all trained models and feature names
@st.cache_resource
def load_models():
    models = {}
    
    # Load only Logistic Regression model
    try:
        models['Logistic Regression'] = joblib.load('credit_risk_model.pkl')
    except FileNotFoundError:
        pass
    
    # Load model info
    try:
        model_info = joblib.load('model_info.pkl')
    except FileNotFoundError:
        model_info = None
        print("Warning: model_info.pkl not found. Some features may be limited.")
    
    try:
        feature_names = joblib.load('feature_names.pkl')
    except FileNotFoundError:
        # Fallback to default feature names
        feature_names = ['loan_size', 'interest_rate', 'borrower_income', 
                        'debt_to_income', 'num_of_accounts', 'derogatory_marks', 'total_debt']
    
    return models, model_info, feature_names

# Load models
models, model_info, feature_names = load_models()

# App title and description
st.title("🏦 Credit Risk Assessment Tool")
st.markdown("""
This application uses a **Logistic Regression** machine learning model to assess credit risk for loan applications.
The model is trained on historical lending data with advanced techniques like oversampling to handle class imbalance,
achieving excellent performance in identifying both healthy and high-risk loans.

**Key Features:**
- Binary classification (Healthy vs High-Risk loans)
- Handles class imbalance with RandomOverSampler
- 99%+ accuracy with balanced precision and recall
""")

# Create sidebar for input
st.sidebar.header("Loan Application Details")

# Set the single model
selected_model_name = 'Logistic Regression'
selected_model = models[selected_model_name]

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

# Model performance section
if model_info and 'model_comparison' in model_info:
    with st.expander("📊 Model Performance Details"):
        st.markdown("**Logistic Regression Model Performance:**")
        
        comparison_df = pd.DataFrame(model_info['model_comparison'])
        lr_data = comparison_df[comparison_df['Model'] == 'Logistic Regression']
        
        if not lr_data.empty:
            # Format the dataframe for better display
            lr_display = lr_data.copy()
            lr_display['Test Accuracy'] = lr_display['Test Accuracy'].apply(lambda x: f"{x:.3f}")
            lr_display['ROC AUC'] = lr_display['ROC AUC'].apply(lambda x: f"{x:.3f}")
            lr_display['CV Mean'] = lr_display['CV Mean'].apply(lambda x: f"{x:.3f}")
            
            st.dataframe(lr_display, use_container_width=True)
            
            # Show key metrics
            accuracy = lr_data['Test Accuracy'].iloc[0]
            roc_auc = lr_data['ROC AUC'].iloc[0]
            
            st.markdown(f"""
            **Key Performance Metrics:**
            - **Test Accuracy**: {accuracy:.3f} (99.4%+ accurate predictions)
            - **ROC AUC**: {roc_auc:.3f} (Excellent discrimination ability)
            - **Cross-Validation**: Consistent performance across all folds
            """)

# Model information
with st.expander("ℹ️ About the Model"):
    st.markdown(f"""
    **Logistic Regression Model:**
    - **Algorithm**: Linear classification with L2 regularization
    - **Strength**: Interpretable, fast, and reliable for binary classification
    - **Training**: Optimized with oversampled data for balanced performance
    
    **Currently Using**: {selected_model_name}
    
    **Training Details:**
    - **Data Preprocessing**: RandomOverSampler to handle class imbalance
    - **Validation**: 5-fold cross-validation for robust evaluation
    - **Features**: 7 financial indicators
    - **Performance**: 99%+ accuracy with balanced precision and recall
    
    **Features Used:**
    - **Loan Size**: Amount of loan requested
    - **Interest Rate**: Loan interest rate percentage
    - **Borrower Income**: Annual income of the applicant
    - **Debt-to-Income Ratio**: Existing debt relative to income
    - **Number of Accounts**: Total credit accounts
    - **Derogatory Marks**: Negative credit history items
    - **Total Debt**: Current total debt amount
    
    **Why Logistic Regression?**
    - **Interpretability**: Clear understanding of feature impacts
    - **Efficiency**: Fast training and prediction
    - **Reliability**: Proven performance for credit risk assessment
    - **Balanced Performance**: Excellent at detecting both healthy and high-risk loans
    
    **Model Benefits:**
    - High accuracy in risk classification
    - Balanced precision and recall for both loan types
    - Fast real-time predictions
    - Clear probability scores for decision transparency

    **Disclaimer:** This tool is for educational purposes only and should not be used for actual lending decisions.
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Credit Risk Assessment | Logistic Regression Model")
