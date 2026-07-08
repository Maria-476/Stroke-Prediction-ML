# 🧠 Stroke Risk Prediction System

## Overview
An end-to-end Machine Learning project that predicts stroke risk 
based on patient health indicators such as glucose level, 
hypertension, age, and BMI.

## Dataset
- **Source:** Healthcare Stroke Prediction Dataset
- **Size:** 4909 rows, 11 features
- **Type:** Binary Classification (Stroke / No Stroke)
- **Challenge:** Heavily imbalanced dataset (~79% No Stroke, ~21% Stroke)

## Workflow
1. Exploratory Data Analysis (EDA)
2. Bivariate analysis with stroke rate per group
3. Correlation heatmap
4. Preprocessing and label encoding
5. Model training and comparison
6. Streamlit app deployment

## Models Trained & Compared
| Model | Accuracy | Recall | F1 Score |
|---|---|---|---|
| Logistic Regression | 0.92 | 0.86 | 0.82 |
| Decision Tree | 0.95 | 0.88 | 0.88 |
| Random Forest | 0.97 | 0.87 | 0.93 |

## Model Selection Note
Decision Tree was initially selected for deployment due to highest 
Recall (0.88). However, during app testing it showed unrealistic 
binary predictions — classifying purely based on glucose threshold 
(>=140 = Stroke, else No Stroke), ignoring all other features.

Random Forest was selected instead as it uses 100 trees across all 
features, producing realistic probability scores and considering 
hypertension, age, BMI alongside glucose level.

## Key Findings
- avg_glucose_level is the strongest predictor (0.57 importance score)
- Hypertension increases stroke rate from 15% to 79%
- Heart disease patients have 56% stroke rate vs 19% without
- Gender and residence type show no significant impact

## Deployed Model
- **Algorithm:** Random Forest
- **Recall:** 0.87 | **F1:** 0.93 | **Accuracy:** 0.97
- **Imbalance handling:** class_weight='balanced'

## Live App
[🚀 Click here to open the app](https://github.com/Maria-476/Stroke-Prediction-ML.git)

## Libraries
pandas, numpy, matplotlib, seaborn, scikit-learn, streamlit, joblib

## Developer
Maria Anwar | Educational Project | 2026