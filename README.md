## Project 25:-

# 📈 Pakistan Stock Exchange (PSX) Financial Data Science & Predictive Modeling
**Explainable Machine Learning Pipeline for Equity Price Forecasting (AHL vs. LUCKY)**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pakistan-stock-exchange-psx-predictive-modeling-shap-analysis.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn 1.6.1](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 Executive Overview & Live Application
Welcome to the official repository of **Project #25: PSX Financial Data Science Intelligence**. This project builds an end-to-end, audit-ready, and reproducible quantitative research pipeline tailored for the **Pakistan Stock Exchange (PSX)**. Focusing on two major equities—**Arif Habib Limited (AHL)** and **Lucky Cement (LUCKY)**—this system addresses complex financial realities such as corporate actions, data leakage prevention, regularized regression modeling, and game-theoretic model explainability.

Explore the fully deployed interactive web application here:
🔗 **[PSX Intelligence Live Streamlit App](https://pakistan-stock-exchange-psx-predictive-modeling-shap-analysis.streamlit.app/)**

---

## 🛠️ Key Methodological Highlights & Workflow

### 1. Corporate Action Adjustments (Lucky Cement 5:1 Stock Split)
Financial time series are frequently distorted by corporate events. On **April 21, 2025**, Lucky Cement executed a **5-for-1 stock split**, dividing shares to lower nominal prices. 
* **The Problem:** Without intervention, machine learning algorithms and cumulative return graphs misinterpret this structural adjustment as a massive 80% market crash.
* **The Solution:** We implemented retrospective price scaling across the dataset using a boolean mask:
  $$\text{Adj\_Close} = \begin{cases} \frac{\text{Close}}{5} & \text{if } \text{Ticker} = \text{LUCKY} \text{ and } \text{Date} < 2025-04-21 \\ \text{Close} & \text{otherwise} \end{cases}$$
  This normalizes historical trajectories, ensuring continuous, undistorted growth curves (+200% for LUCKY, +350% for AHL).

### 2. Leak-Proof Pipeline Architecture
Time-series data is highly sensitive to information leakage. Standard random train-test splits allow future data points to contaminate past training records.
* **Chronological Splitting:** We strictly segregated the data chronologically (80% training, 20% testing) to simulate real-world forward-looking deployment.
* **Scikit-Learn `Pipeline` & `ColumnTransformer`:** Imputation, scaling (`StandardScaler`), and categorical encoding (`OneHotEncoder`) were encapsulated directly inside cross-validation folds, guaranteeing zero data leakage.

---

## 📐 Mathematical Intuition & Why Lasso Regression?

We selected **Lasso Regression (Least Absolute Shrinkage and Selection Operator)** as our core predictive model. In financial data science, multicollinearity and noisy momentum indicators often lead to overfitting in Ordinary Least Squares (OLS) regression.

### The Lasso Optimization Objective
Lasso introduces an $L_1$-norm regularization penalty to the residual sum of squares (RSS), forcing less important feature coefficients to absolute zero for automatic feature selection:

$$\min_{\beta} \left( \frac{1}{2n} \sum_{i=1}^{n} (y_i - x_i^T \beta)^2 + \alpha \sum_{j=1}^{p} \vert{}\beta_j\vert{} \right)$$

Where:
* $y_i$ is the target closing price.
* $x_i$ represents our engineered features (`Volume`, `Prev_Rate`, `Rolling_Mean_5`, `Ticker`).
* $\alpha$ is the regularization hyperparameter tuned via `GridSearchCV`.
* $\beta_j$ represents feature coefficients.

### Hyperparameter Tuning Results
Using 3-fold cross-validation (`cv=3`) across an $\alpha$ grid ranging from $0.001$ to $100.0$, the optimal parameter was identified at **$\alpha = 1.0$**, achieving exceptional predictive stability:
* **Best Cross-Validation $R^2$ Score:** **0.9893**
* **Test Set $R^2$ Score:** **0.9716**

---

## 🔍 Model Explainability via SHAP (SHapley Additive exPlanations)

To ensure institutional trust and avoid black-box ambiguity, we integrated **SHAP LinearExplainer** to calculate exact cooperative game-theoretic feature attributions.

### Mathematical Attribution
Shapley values assign a marginal contribution weight $\phi_i$ to each feature $i$ across all possible feature subsets $S$:

$$\phi_i (v) = \sum_{S \subseteq \setminus \{i\}} \frac{\vert{}S\vert{}! (p - \vert{}S\vert{} - 1)!}{p!} \left( v(S \cup \{i\}) - v(S) \right)$$

### Key Explanatory Findings:
1. **`num__Prev_Rate` Dominance:** Yesterday's closing price acts as the ultimate anchor, driving over **88%** of the model's absolute price output.
2. **Momentum Filtering:** `Rolling_Mean_5` and trading volume provide essential short-term trend context, correcting for daily volatility.

---

## 📁 Repository Directory Structure

```text
📦 pakistan-stock-exchange-psx-predictive-modeling-shap-analysis
 ┣ 📜 app.py                     # Streamlit interactive web application
 ┣ 📜 psx_lasso_pipeline.pkl      # Serialized Scikit-Learn model pipeline (Joblib)
 ┣ 📜 requirements.txt            # Pinned exact environment dependencies
 ┗ 📜 README.md                   # Project documentation and mathematical guide
```

## 🚀 Local Installation and Quick Start:-
**Clone the repository:**
git clone [https://github.com/M-Nafay-Ali/pakistan-stock-exchange-psx-predictive-modeling-shap-analysis.git](https://github.com/M-Nafay-Ali/pakistan-stock-exchange-psx-predictive-modeling-shap-analysis.git)
cd pakistan-stock-exchange-psx-predictive-modeling-shap-analysis
**Install Dependencies**
pip install -r requirements.txt
**Run the Streamlit application locally**
streamlit run app.py


##  🧩 Tech Stack & Dependencies:-
**Core Language:** Python 3.10+
**Machine Learning:** Scikit-Learn 1.6.1 (Pipeline, ColumnTransformer, Lasso, GridSearchCV)
**Model Serialization:** Joblib 1.4.2
**Data Manipulation:** Pandas 2.2.3, NumPy 2.2.2
**Data Visualization:** Plotly 5.24.1, Matplotlib, Seaborn
**Model Interpretability:** Shap
**Web Application Framework:** Streamlit 1.42.0

## 📞 Contact Information:-
* **Email:-**[englandengland271@gmail.com]
* **Linkedin:-**[https://www.linkedin.com/in/mohammed-nafay-ali-16519138a?utm_source=share_via&utm_content=profile&utm_medium=member_android]
* **GitHub:-**[https://github.com/M-Nafay-Ali]


 
