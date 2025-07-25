# Credit Risk Classification using Multiple ML Models

## Overview
This project implements a comprehensive credit risk assessment system using multiple machine learning algorithms to classify loan applications as 'healthy' or 'high-risk'. The system compares the performance of individual models (Logistic Regression, Random Forest, XGBoost) and ensemble methods, with a user-friendly Streamlit web application for real-time predictions.

## Key Features
- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, and Ensemble methods
- **Advanced Data Handling**: Oversampling techniques to address class imbalance
- **Model Comparison**: Comprehensive performance evaluation and benchmarking
- **Interactive Web App**: Streamlit-based interface for real-time credit risk assessment
- **Production Ready**: Model serialization and deployment-ready architecture

## Dataset
The analysis uses lending data with 77,536 loan records containing:
- **Features**: 7 financial indicators (loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks, total debt)
- **Target**: Binary classification (0 = healthy loan, 1 = high-risk loan)
- **Class Distribution**: Highly imbalanced dataset (96.7% healthy vs 3.3% high-risk loans)

## Methodology

### 1. Data Preprocessing
- **Class Imbalance Handling**: Applied RandomOverSampler to balance the dataset
- **Train-Test Split**: 75% training, 25% testing with stratified sampling
- **Feature Scaling**: Standardized features for optimal model performance

### 2. Model Implementation
- **Logistic Regression**: Baseline linear model with L2 regularization
- **Random Forest**: Ensemble of 100 decision trees with optimized hyperparameters
- **XGBoost**: Gradient boosting classifier for complex pattern recognition
- **Ensemble Methods**: Soft Voting Classifier combining all three models

### 3. Model Evaluation
- **Cross-Validation**: 5-fold CV for robust performance estimation
- **Metrics**: Accuracy, ROC AUC, Precision, Recall, F1-Score
- **Comprehensive Comparison**: Performance benchmarking across all models

## Results

### Model Performance Comparison
| Model | Test Accuracy | ROC AUC | CV Mean | CV Std |
|-------|---------------|---------|---------|---------|
| Random Forest | 99.53% | 0.9994 | 0.9954 | 0.0004 |
| Soft Voting (LR+RF+XGB) | 99.49% | 0.9985 | 0.9953 | 0.0005 |
| XGBoost | 99.47% | 0.9980 | 0.9950 | 0.0005 |
| Logistic Regression | 99.42% | 0.9945 | 0.9947 | 0.0006 |

### Key Insights
- **Best Overall Model**: Random Forest achieved the highest performance metrics
- **Ensemble Benefits**: Soft voting ensemble matched individual best performance with increased robustness
- **Class Balance Impact**: Oversampling significantly improved high-risk loan detection (from 85% to 99% precision)
- **Feature Importance**: Debt-to-income ratio and total debt were the most predictive features

## Technical Implementation

### Requirements
```
streamlit
pandas
scikit-learn
joblib
numpy
xgboost
imbalanced-learn
matplotlib
```

## Streamlit Web Application

The project includes a production-ready web application with:
- **Model Selection**: Choose between individual models or ensemble methods
- **Real-time Predictions**: Interactive input form for loan application data
- **Performance Metrics**: Display of model accuracy and confidence scores
- **Risk Assessment**: Visual probability breakdown and risk classification
- **Model Comparison**: Side-by-side performance comparison of all models

### Running the Application
```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## Model Deployment

All trained models are serialized using joblib for easy deployment:
- Individual models saved separately for comparison
- Best performing model identified and saved for production use
- Model metadata and performance metrics stored for monitoring
- Feature names preserved for consistent data preprocessing

## Business Impact

### Risk Management
- **High-Risk Detection**: 99% precision in identifying risky loans
- **False Positive Reduction**: Minimized healthy loan rejections
- **Cost Savings**: Improved decision-making reduces potential loan defaults

### Model Interpretability
- **Feature Importance**: Clear understanding of key risk factors
- **Confidence Scores**: Transparency in prediction certainty
- **Multiple Perspectives**: Ensemble approach provides robust predictions

## Future Enhancements

1. **Advanced Features**: Incorporate additional financial indicators
2. **Real-time Data**: Integration with live credit bureau APIs
3. **Model Monitoring**: Automated performance tracking and retraining
4. **Explainable AI**: SHAP values for individual prediction explanations
5. **A/B Testing**: Framework for testing new model improvements

## License
This project is for educational and demonstration purposes.

## Contact
For questions or collaboration opportunities, please feel free to reach out!
