# 🎓 Student Dropout Prediction & Analysis Dashboard

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-ff69b4.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

readme = r"""# EduShield AI

## Early Student Dropout Risk Detection and Support System

EduShield AI is a machine-learning-based decision-support system designed to identify students who may be at risk of academic failure or dropping out at an early stage.

The system analyses academic and engagement-related factors such as attendance, internal marks, assignment completion, previous backlogs and other relevant indicators. It produces an estimated risk level and helps authorized faculty members understand important contributing factors and consider suitable academic support.

> **Important:** EduShield AI is an early-warning and decision-support system. It does not confirm that a student will drop out, and it should not be used for automatic disciplinary decisions. Final intervention decisions remain with authorized faculty members/counsellors.


An interactive web application built with Streamlit to predict student dropout, analyze contributing factors, and provide actionable insights using machine learning and model explainability techniques.
## Problem Statement

Educational institutions may identify academically struggling students only after poor examination results, severe attendance shortages, repeated backlogs or disengagement have already occurred.

Late identification reduces the time available for meaningful intervention.

EduShield AI addresses this problem by providing an early-warning mechanism that analyses available student academic and engagement data and presents a risk estimate to authorized faculty members.

---

## Objectives

1. Identify potentially at-risk students at an early stage.
2. Analyse multiple academic and engagement indicators together.
3. Generate an interpretable student risk score.
4. Classify students into Low, Medium and High Risk categories.
5. Highlight important factors contributing to the prediction.
6. Provide support-oriented recommendations for faculty.
7. Provide class-level analytics through a web dashboard.
8. Evaluate multiple machine-learning models using appropriate metrics.
9. Maintain a privacy-conscious and human-in-the-loop approach.

---

## Proposed Solution

EduShield AI follows this general pipeline:

```text
Student Academic / Engagement Data
                |
                v
        Data Validation
                |
                v
      Data Cleaning & Preprocessing
                |
                v
       Exploratory Data Analysis
                |
                v
          Feature Engineering
                |
                v
        Machine Learning Model
                |
                v
          Risk Probability
                |
                v
       Low / Medium / High Risk
                |
                v
       Explainable Risk Factors
                |
                v
     Faculty Support Recommendations
                |
                v
       Dashboard / Follow-up

Support Recommendations

EduShield AI is designed to support intervention, not punishment.

Example recommendation logic:

If attendance is low:
    Recommend attendance counselling.


If academic marks are low:
    Recommend remedial academic support.


If assignments are incomplete:
    Recommend assignment/subject mentoring.


If financial difficulty is indicated:
    Recommend checking available institutional support.


If multiple risk factors are present:
    Recommend faculty mentor/counsellor review.

Recommendations should remain configurable and should not be presented as guaranteed solutions.

## 🌟 Key Features

This dashboard provides a complete, end-to-end workflow for student dropout analysis:

### 📊 Data Overview
Get a quick summary of the dataset, including data quality checks, student demographics, academic performance, and key risk factors.

### 📈 Exploratory Data Analysis (EDA)
Interactively explore feature distributions, correlations, and their relationship with student outcomes (Dropout, Graduate, Enrolled).

### 🤖 Model Training
Train a Random Forest Classifier with a single click to predict student outcomes.

### ✅ Model Evaluation
Assess model performance using accuracy, classification reports, and an interactive confusion matrix.

### 🧠 Model Explainability (XAI)
- **Global Explanations**: Understand the most important features driving predictions across the entire dataset using SHAP, Permutation Importance, and built-in feature importance.
- **Local Explanations**: Dive deep into why the model made a specific prediction for an individual student using SHAP Waterfall Plots and LIME.

### 🔮 Individual Prediction Tool
- Input a student's data using interactive sliders and dropdowns
- Receive an instant prediction of the student's likely outcome (Dropout, Graduate, or Enrolled)
- Get a detailed explanation of the factors that influenced the prediction, along with actionable recommendations

### 🔧 Interactive Feature Analysis
- Explore how changing a single feature's value impacts the model's prediction probabilities
- Use the interactive feature explorer to view detailed statistics and dropout rates for any column

### 📂 Custom Data Upload
Upload your own student dataset in CSV format to use the dashboard's full capabilities.

## 📸 Screenshots

| Data Overview & EDA | Model Explainability (SHAP) | Individual Prediction with Explanation |
|:---:|:---:|:---:|
| ![EDA Screenshot](path/to/eda-screenshot.png) | ![SHAP Screenshot](path/to/shap-screenshot.png) | ![Prediction Screenshot](path/to/prediction-screenshot.png) |

*Replace the image links above with actual screenshots of your running application.*

## 🛠️ Technology Stack

- **Framework**: Streamlit
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Data Visualization**: Matplotlib, Seaborn, Plotly
- **Model Explainability**: SHAP, LIME

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

### Running the Application

1. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
   *(Replace `app.py` with the actual name of your Python script if it's different.)*

2. **Open your web browser:**
   Navigate to `http://localhost:8501`. The application should now be running.

## 📋 How to Use the Dashboard

The dashboard is organized into four main modules accessible from the sidebar navigation:

### 1. Data Overview
Start here to get a high-level understanding of your dataset.

### 2. Exploratory Data Analysis
Dive deeper into the data. Use the interactive charts to uncover trends and relationships between different student attributes and their final outcomes.

### 3. Model Training & Evaluation
- Click the "Start Training" button to build the prediction model
- Once trained, view the model's performance metrics and feature importance charts
- Explore the "Model Explainability" tab to understand how the model works on a global and local level

### 4. Dropout Prediction
- Navigate to this section to use the interactive prediction tool
- Adjust the sliders and inputs to match a student's profile
- Click "Predict with Explanation" to see the predicted outcome and the key factors that led to that decision

## 💾 Data

The application comes pre-loaded with a sample dataset (`student_dropout_data.csv`) from the UCI Machine Learning Repository. This dataset contains various demographic, socio-economic, and academic features for students.

You can also upload your own CSV file using the file uploader in the sidebar. Ensure your dataset has a `Target` column with values like 'Dropout', 'Graduate', and 'Enrolled' for full functionality.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or find any issues, please feel free to:

1. Fork the repository
2. Create a new feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📦 Requirements

Create a `requirements.txt` file in your repository with the following content:

```txt
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
shap
lime
plotly
```

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the student dataset
- Streamlit community for the excellent framework
- SHAP and LIME libraries for model explainability

## Limitations

A prediction model cannot guarantee that a student will drop out.

Model quality depends strongly on dataset quality and representativeness.

Public or synthetic datasets may not represent a particular institution.

Predictions can contain false positives and false negatives.

Risk thresholds require validation.

Some student factors may be unavailable or difficult to measure.

Real-world deployment requires institutional approval, privacy controls and responsible governance.

## Team

Project: EduShield AI
Category: Machine Learning / Digital Learning / Smart Education
Application: Early Student Risk Detection and Academic Support


1. Nitesh Yadav — Machine Learning & Data Science Lead
2. Makani Manthan — Application & Dashboard Developer
3. Sumit Nakum — Research, Testing & Documentation Lead


