# Disease Prediction from Symptoms

This is a Streamlit-based Machine Learning web application that predicts possible diseases based on natural language descriptions of symptoms. It leverages classification models including Random Forest, Support Vector Machine (SVM), and Logistic Regression, trained on a Kaggle dataset.

---

## Features

- Input symptoms in plain English (e.g., *"I have a sore throat and fever for the last 3 days."*)
- Automatically extracts keywords using custom preprocessing
- Trains and compares three ML models: Random Forest, SVM, and Logistic Regression
- Predicts top probable diseases with explanation of matching keywords
- Displays symptom examples and information about each predicted disease
- Visualizes:
  - Model accuracy
  - Confusion matrices
  - Feature importance
  - Dataset disease distribution
  - Symptom keyword frequency

---

## Technologies Used

- **Python**
- **Streamlit** – web UI
- **scikit-learn** – machine learning models
- **pandas, numpy** – data handling
- **plotly, seaborn, matplotlib** – visualizations
- **joblib** – model saving

---


---

##  How to Run

1. Install the required packages:
```bash
pip install streamlit scikit-learn pandas numpy matplotlib seaborn plotly joblib


## Dataset

- **Source**: Kaggle  
- **Format**: CSV  
- **Description**: Natural language symptom descriptions mapped to disease labels

---

##  Machine Learning Models

The application trains and evaluates the following models:

- Random Forest Classifier  
- Support Vector Machine (SVM)  
- Logistic Regression

Each model is evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1-Score




