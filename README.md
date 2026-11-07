# AI-Powered Forest Cover Type Classification

## Overview
Machine learning project to predict forest cover types using the UCI Covertype dataset.

## Dataset
- 581,012 samples
- 54 features
- 7 forest cover classes

## Models and Results

| Model | Test Accuracy | Test Macro F1 |
|---|---:|---:|
| Extra Trees | 93.20% | 0.8933 |
| Random Forest | 92.30% | 0.8735 |
| Stacking Ensemble | 92.15% | 0.8625 |
| Decision Tree | 82.87% | 0.7448 |

## Technologies
Python, Scikit-learn, Pandas, NumPy, Matplotlib, Streamlit, Joblib.

## Project Structure
- notebooks/: Jupyter notebook
- app/: Streamlit application
- models/: Trained machine learning models
- results/: Evaluation metrics and graphs

## Run Application
Install dependencies:
pip install -r requirements.txt

Run:
streamlit run app/app.py

## Evaluation
Accuracy, Macro F1-score, Confusion Matrix, Training Time and Inference Time.

## Purpose
Academic machine learning project.
