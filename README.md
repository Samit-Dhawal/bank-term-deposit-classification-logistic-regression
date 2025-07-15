# Bank Term Deposit Classification - Logistic Regression

A machine learning project that predicts whether a bank client will subscribe to a term deposit based on client demographics, contact information, and previous campaign data using logistic regression classification.

## 🎯 Project Overview

**Bank Term Deposit Classification - Python, Machine Learning, Logistic Regression, Scikit-learn**

* **Challenge**: A bank required an automated system to predict client subscription likelihood for term deposits based on demographic data, contact history, and previous campaign outcomes to optimize marketing strategies.

* **Solution**: Engineered a logistic regression classification model with comprehensive feature analysis and correlation study, implementing advanced data preprocessing and model evaluation techniques.

* **Impact**: Achieved 89% accuracy with ROC-AUC score of 0.57, enabling targeted marketing campaigns and reducing customer acquisition costs through data-driven client segmentation.

## 📊 Dataset Overview

The project utilizes a comprehensive bank marketing dataset with 17 attributes including:

### Client Demographics
- **Age**: Numeric client age
- **Job**: Professional category (12 categories)
- **Marital Status**: Relationship status
- **Education**: Educational background level
- **Default**: Credit default status
- **Balance**: Average yearly balance in euros

### Contact Information
- **Housing**: Housing loan status
- **Loan**: Personal loan status
- **Contact**: Communication type (telephone/cellular)
- **Day/Month**: Last contact timing
- **Duration**: Contact duration in seconds

### Campaign Data
- **Campaign**: Number of contacts in current campaign
- **Pdays**: Days since last contact from previous campaign
- **Previous**: Previous campaign contact count
- **Poutcome**: Previous campaign outcome

### Target Variable
- **y**: Term deposit subscription (binary: "yes"/"no")

## 🛠️ Technology Stack

- **Python**: Core programming language
- **Scikit-learn**: Machine learning library for logistic regression
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization and correlation analysis
- **Jupyter Notebook**: Interactive development environment

## 📈 Model Performance

### Classification Metrics
- **Overall Accuracy**: 89%
- **ROC-AUC Score**: 0.57
- **Precision (Class 0)**: 0.90
- **Recall (Class 0)**: 0.98
- **F1-Score (Class 0)**: 0.94
- **Precision (Class 1)**: 0.57
- **Recall (Class 1)**: 0.16
- **F1-Score (Class 1)**: 0.25

### Model Insights
- High precision for predicting non-subscribers (Class 0)
- Strong recall for identifying actual non-subscribers
- Challenging class imbalance with term deposit subscribers being minority class
- ROC curve analysis shows moderate discriminative ability

## 🏗️ Project Structure

```
bank-term-deposit-classification-logistic-regression/
├── Logistic_Regression_Samit.ipynb    # Main analysis notebook
├── bank-full.csv                      # Complete dataset
├── bank-names.txt                     # Feature descriptions
├── Problem-Statement(Bank_data).txt   # Project requirements
└── README.md                          # Project documentation
```

## 🔍 Key Features

### Data Analysis & Preprocessing
- **Correlation Analysis**: Comprehensive heatmap visualization of feature relationships
- **Feature Engineering**: Categorical variable encoding and numerical scaling
- **Data Cleaning**: Handling missing values and outlier detection
- **Class Balance Assessment**: Analysis of target variable distribution

### Model Development
- **Logistic Regression Implementation**: Binary classification with sklearn
- **Feature Selection**: Correlation-based feature importance analysis
- **Model Validation**: Train-test split with performance evaluation
- **Hyperparameter Tuning**: Optimization for best classification performance

### Evaluation & Visualization
- **ROC Curve Analysis**: Visual assessment of model discriminative power
- **Confusion Matrix**: Detailed classification performance breakdown
- **Classification Report**: Precision, recall, and F1-score metrics
- **Feature Correlation Heatmap**: Visual representation of variable relationships

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- Jupyter Notebook or JupyterLab
- Required Python packages (see installation)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Samit-Dhawal/bank-term-deposit-classification-logistic-regression.git
   cd bank-term-deposit-classification-logistic-regression
   ```

2. **Install required packages**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook Logistic_Regression_Samit.ipynb
   ```

## 📊 Usage

### Running the Analysis

1. **Open the main notebook**: `Logistic_Regression_Samit.ipynb`
2. **Execute cells sequentially** to:
   - Load and explore the dataset
   - Perform correlation analysis
   - Preprocess the data
   - Train the logistic regression model
   - Evaluate model performance
   - Generate visualizations

### Key Outputs
- **Correlation heatmap** showing feature relationships
- **ROC curve** with AUC score
- **Classification report** with detailed metrics
- **Model predictions** on test dataset

## 🎯 Business Applications

### Marketing Strategy Optimization
- **Target Identification**: Focus marketing efforts on high-probability clients
- **Resource Allocation**: Optimize campaign budget based on prediction confidence
- **Customer Segmentation**: Categorize clients by subscription likelihood

### Risk Assessment
- **Campaign Planning**: Predict success rates for different client segments
- **Contact Strategy**: Optimize timing and frequency of client outreach
- **ROI Improvement**: Increase conversion rates through data-driven targeting

## 📈 Model Insights

### Key Findings
- **Strong performance** in identifying non-subscribers (98% recall)
- **Class imbalance challenges** with term deposit subscribers being minority
- **Feature correlations** reveal important predictive relationships
- **Contact duration** and previous campaign outcomes show significance

### Limitations
- **Low recall for subscribers** (16%) indicates missed opportunities
- **Moderate AUC score** suggests room for advanced modeling techniques
- **Class imbalance** requires additional balancing strategies

## 🔮 Future Enhancements

- **Advanced Algorithms**: Implement Random Forest, XGBoost, or Neural Networks
- **Feature Engineering**: Create interaction terms and polynomial features
- **Class Balancing**: Apply SMOTE or other resampling techniques
- **Hyperparameter Optimization**: Grid search and cross-validation
- **Ensemble Methods**: Combine multiple models for improved performance
- **Real-time Prediction**: Deploy model for live client scoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- Dataset provided by UCI Machine Learning Repository
- Scikit-learn community for machine learning tools
- Python data science ecosystem contributors

## 📞 Contact

**Samit Dhawal**
- GitHub: [@Samit-Dhawal](https://github.com/Samit-Dhawal)
- Project Link: [https://github.com/Samit-Dhawal/bank-term-deposit-classification-logistic-regression](https://github.com/Samit-Dhawal/bank-term-deposit-classification-logistic-regression)

---

**Built with Python, Scikit-learn, and Machine Learning expertise**
