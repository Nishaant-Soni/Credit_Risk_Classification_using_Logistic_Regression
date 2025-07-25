# Credit Risk Classification using Logistic Regression

## Overview
This project implements a credit risk assessment system using **Logistic Regression** to classify loan applications as 'healthy' or 'high-risk'. The model uses advanced data preprocessing techniques including oversampling to handle class imbalance, achieving excellent performance with 99%+ accuracy. The system includes a user-friendly Streamlit web application for real-time credit risk assessment.

## Key Features
- **Logistic Regression Model**: Interpretable and reliable binary classification
- **Advanced Data Handling**: RandomOverSampler to address severe class imbalance
- **High Performance**: 99%+ accuracy with balanced precision and recall
- **Interactive Web App**: Streamlit-based interface for real-time predictions
- **Production Ready**: Complete model serialization and deployment architecture

## Dataset
The analysis uses lending data with 77,536 loan records containing:
- **Features**: 7 financial indicators (loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks, total debt)
- **Target**: Binary classification (0 = healthy loan, 1 = high-risk loan)
- **Class Distribution**: Highly imbalanced dataset (96.7% healthy vs 3.3% high-risk loans)

## Methodology

## Methodology

### 1. Data Preprocessing
- **Class Imbalance Handling**: Applied RandomOverSampler to balance the dataset
- **Train-Test Split**: 75% training, 25% testing with stratified sampling
- **Feature Standardization**: Optimized feature scaling for logistic regression

### 2. Model Implementation
- **Algorithm**: Logistic Regression with L2 regularization
- **Optimization**: Trained on balanced dataset using oversampling
- **Hyperparameters**: Default scikit-learn parameters with random_state=1 for reproducibility

### 3. Model Evaluation
- **Cross-Validation**: 5-fold CV for robust performance estimation
- **Metrics**: Accuracy, ROC AUC, Precision, Recall, F1-Score
- **Performance Analysis**: Detailed classification reports and confusion matrices

## Results

### Model Performance
| Metric | Value | Description |
|--------|-------|-------------|
| Test Accuracy | 99.42% | Overall prediction accuracy |
| ROC AUC | 0.9945 | Excellent discrimination ability |
| CV Score | 0.9947 ± 0.0006 | Consistent cross-validation performance |
| Precision (Healthy) | 99% | Accuracy in predicting healthy loans |
| Precision (High-Risk) | 99% | Accuracy in predicting high-risk loans |
| Recall (Healthy) | 99% | Detection rate of healthy loans |
| Recall (High-Risk) | 99% | Detection rate of high-risk loans |

### Key Insights
- **Excellent Performance**: 99%+ accuracy across all key metrics
- **Balanced Classification**: Equal precision and recall for both loan types
- **Class Balance Impact**: Oversampling dramatically improved high-risk loan detection
- **Model Reliability**: Consistent performance across cross-validation folds
- **Interpretability**: Clear feature coefficients for business understanding

## Technical Implementation

### Requirements
```
streamlit
pandas
scikit-learn
joblib
numpy
imbalanced-learn
matplotlib
```

## Streamlit Web Application

The project includes a production-ready web application featuring:
- **Real-time Predictions**: Interactive input form for loan application data
- **Performance Metrics**: Display of model accuracy and confidence scores
- **Risk Assessment**: Visual probability breakdown and risk classification
- **Model Details**: Comprehensive information about the Logistic Regression model
- **Responsive Design**: Clean, professional interface for easy use

### Running the Application
```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## Model Deployment

The trained Logistic Regression model is serialized using joblib for easy deployment:
- Model saved with optimal hyperparameters and training data
- Model metadata and performance metrics stored for monitoring
- Feature names preserved for consistent data preprocessing
- Complete deployment package for production use

## Business Impact

### Risk Management
- **High-Risk Detection**: 99% precision in identifying risky loans
- **Balanced Performance**: Equal accuracy for both healthy and high-risk loans
- **Cost Savings**: Improved decision-making reduces potential loan defaults
- **Fast Processing**: Real-time predictions for immediate loan decisions

### Model Advantages
- **Interpretability**: Clear understanding of feature impacts on risk
- **Efficiency**: Fast training and prediction capabilities
- **Reliability**: Proven algorithm with consistent performance
- **Transparency**: Clear probability scores for decision justification

## Why Logistic Regression?

1. **Interpretability**: Linear coefficients provide clear feature importance
2. **Efficiency**: Fast training and prediction for real-time applications
3. **Reliability**: Well-established algorithm with proven performance
4. **Balanced Performance**: Excellent results for both loan categories
5. **Regulatory Compliance**: Transparent model suitable for financial regulations

## Future Enhancements

1. **Feature Engineering**: Incorporate additional financial indicators
2. **Real-time Integration**: Connect with live credit bureau APIs
3. **Model Monitoring**: Automated performance tracking and alerts
4. **Threshold Optimization**: Dynamic probability thresholds based on business needs
5. **Reporting Dashboard**: Advanced analytics and reporting capabilities

## License
This project is for educational and demonstration purposes.

## Contact
For questions or collaboration opportunities, please feel free to reach out!
