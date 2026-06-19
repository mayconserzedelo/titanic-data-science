# Titanic Survival Analysis

> A polished, portfolio-ready Data Science project exploring the factors behind passenger survival on the Titanic using Python, pandas, NumPy, visualization, and machine learning.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-data%20analysis-150458.svg)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-machine%20learning-F7931E.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

---

## Overview

This project analyzes the classic Titanic dataset to answer a simple but powerful question:

**Which passenger characteristics were most strongly associated with survival?**

The repository is structured as a complete end-to-end Data Science case study, from data cleaning and exploratory analysis to feature engineering, model training, and evaluation.

---

## Why this project?

The Titanic dataset is a well-known introduction to classification problems, but it becomes much more valuable when presented as a professional portfolio project.

This version is designed to demonstrate:

- Strong data analysis fundamentals
- Clear business/problem framing
- Practical Python and pandas usage
- Statistical thinking and visualization
- Basic machine learning workflow
- Clean repository organization
- Communication of insights in a recruiter-friendly format

---

## Project goals

- Explore the structure and quality of the Titanic dataset
- Identify patterns related to passenger survival
- Handle missing values and categorical variables
- Engineer useful features for modeling
- Train baseline classification models
- Compare model performance using standard metrics
- Present clear conclusions and next steps

---

## Key questions

- Did women survive at higher rates than men?
- Did passenger class affect survival probability?
- Was age an important factor?
- Did family size or ticket fare influence the outcome?
- Can survival be predicted with reasonable accuracy using simple models?

---

## Repository structure

```bash
titanic-data-science/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── titanic_analysis_roadmap.ipynb
├── models/
├── reports/
│   └── figures/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── modeling.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Workflow

### 1. Data understanding
Inspect the Titanic dataset, variable types, target distribution, and missing values.

### 2. Data cleaning
Standardize column names, handle nulls, and remove or simplify low-value columns.

### 3. Exploratory Data Analysis
Study survival patterns by sex, class, age, fare, and embarkation point.

### 4. Feature engineering
Create more informative variables such as family size, title, and age bands.

### 5. Modeling
Train baseline classifiers such as Logistic Regression and Random Forest.

### 6. Evaluation
Measure model quality with accuracy, precision, recall, f1-score, and confusion matrix.

---

## Main findings

> Replace these placeholders with your real results after running the analysis.

- Female passengers generally show higher survival rates.
- Higher passenger class is often associated with better survival outcomes.
- Age and fare may add predictive value.
- Feature engineering can improve the clarity and performance of simple baseline models.

---

## Expected deliverables

By the end of the project, this repository should contain:

- A clean and readable notebook
- Reusable Python functions in `src/`
- Visualizations saved in `reports/figures/`
- A trained model or model comparison summary
- A concise explanation of the key insights

---

## How to run the project

### 1. Clone the repository
```bash
git clone https://github.com/mayconserzedelo/titanic-data-science.git
cd titanic-data-science
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

### 3. Activate the environment

**Windows**
```bash
venv\\Scripts\\activate
```

**macOS / Linux**
```bash
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Open the notebook
```bash
jupyter notebook
```

---

## Recommended next improvements

- Add hyperparameter tuning
- Test additional models
- Build a cleaner preprocessing pipeline
- Create a dashboard or interactive visualization
- Add a short written case study in English and Portuguese
- Publish a summary post on LinkedIn

---

## About the author

**Maycon Serzedelo**  
Economics graduate transitioning into Data Science  
Currently studying Python, SQL, ETL, pandas, and NumPy

- GitHub: [mayconserzedelo](https://github.com/mayconserzedelo)
- LinkedIn: https://www.linkedin.com/in/maycon-serzedelo-854972215/
- Email: serzedelomaycon@gmail.com

---

## License

This project is intended for educational and portfolio purposes.
