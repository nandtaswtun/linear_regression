# Amazon Sales Order Value Prediction

## 🚀 Project Overview

This project focuses on building a **Linear Regression model** to accurately predict the total sales amount (`TotalAmount`) of an Amazon order using transactional features. The goal is to understand which variables have the most significant impact on the final order value.

## 💾 Dataset

The analysis uses an Amazon Sales dataset (`Amazon (1).csv`) containing 100,000 order records.

**Key Features Used for Prediction:**
* **Quantitative:** `Quantity`, `UnitPrice`, `Discount`, `Tax`, `ShippingCost`
* **Categorical:** `Category`, `Brand`, `Country`

## ⚙️ Methodology

1.  **Data Preprocessing:** Categorical features (`Category`, `Brand`, `Country`) were converted into numerical features using **One-Hot Encoding** with `drop_first=True` to avoid multicollinearity.
2.  **Data Splitting:** The data was split into a Training set (80%) and a Testing set (20%) using `train_test_split` to ensure the model's performance is evaluated on unseen data.
3.  **Model Training:** A `LinearRegression` model from `scikit-learn` was trained on the training set.
4.  **Evaluation:** The model's performance was assessed using $\text{R}^2$, RMSE, and MAE on the test set.

## ✅ Results and Interpretation

The Linear Regression model demonstrated strong predictive power.

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **R-squared ($\text{R}^2$)** | **0.9095** | **90.95%** of the variability in `TotalAmount` is explained by the features in the model, indicating an excellent fit. |
| **MAE (Mean Absolute Error)** | **\$166.02** | On average, the model's prediction was off by **\$166.02**. |
| **RMSE (Root Mean Squared Error)** | **\$217.11** | The average magnitude of the errors, slightly higher than MAE due to sensitivity to larger prediction errors (outliers). |

## 🛠️ Key Libraries

* `pandas` (Data manipulation)
* `numpy` (Numerical operations)
* `sklearn` (Machine Learning: `LinearRegression`, `train_test_split`, `metrics`)

## 📊 Data Source
https://www.kaggle.com/datasets/rohiteng/amazon-sales-dataset
