# Student Performance Prediction Using Linear Regression

## 1. Abstract
Student Performance Prediction is a Machine Learning project that estimates a student's expected final examination score based on study hours, attendance percentage, and previous examination score.

The project uses **Linear Regression**, a supervised Machine Learning algorithm for predicting continuous numerical values. Python is used for model development, while Streamlit provides a simple web interface.

The application generates a sample dataset, trains the model, evaluates its performance, and displays predicted scores. The project demonstrates the basic workflow of a Machine Learning application.

## 2. Introduction
Machine Learning helps computers identify patterns in data and use those patterns to make predictions. Student performance can be influenced by several factors, including study habits, attendance, and previous academic results. This project demonstrates how these factors can be used as input features to estimate a final examination score.

The project is intended for educational purposes and demonstrates the implementation of a regression model through a web application.

## 3. Problem Statement
Estimating a student's examination score manually can be difficult because multiple factors may influence academic performance. The objective of this project is to develop a simple Machine Learning application that uses study hours, attendance, and previous examination scores to predict the final score.

## 4. Features of the UI
- Sidebar sliders for study hours, attendance, and previous score
- **Predict Score** button with result card, estimated grade, and progress bar
- Model performance tab: R², MAE, RMSE, Actual vs Predicted plot, coefficient chart
- Dataset tab: data table, summary statistics, CSV download
- About tab with a project summary

## 5. Installation and Running
```bash
pip install -r requirements.txt
streamlit run app.py
```
The app opens at http://localhost:8501

## 6. Project Structure
```text
Student-Performance-Prediction/
├── app.py
├── requirements.txt
├── README.md
└── Project_Report.md
```

## 7. Conclusion
The project shows how Linear Regression can predict student performance from a few simple inputs, and how Streamlit can turn a model into an interactive application.
