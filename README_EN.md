# 🛒 Retail Sales Prediction with Machine Learning

> A regression-based predictive model to forecast a retail store's daily sales from calendar and promotion variables, aimed at optimizing inventory management, promotions, and staff scheduling.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white">
  <img alt="pandas" src="https://img.shields.io/badge/pandas-Data%20Wrangling-150458?logo=pandas&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-Computation-013243?logo=numpy&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-Viz-11557C">
  <img alt="seaborn" src="https://img.shields.io/badge/seaborn-Viz-4C72B0">
</p>

> 🇪🇸 *Spanish version available at [README.md](README.md)*

---

## 📑 Table of Contents

- [🛒 Retail Sales Prediction with Machine Learning](#-retail-sales-prediction-with-machine-learning)
  - [📑 Table of Contents](#-table-of-contents)
  - [🎯 Executive Summary](#-executive-summary)
  - [❓ Problem Statement](#-problem-statement)
  - [📊 Dataset](#-dataset)
  - [🔬 Methodology](#-methodology)
  - [📈 Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
    - [Average sales by day of the week](#average-sales-by-day-of-the-week)
    - [Holiday vs. Non-Holiday and Promotions](#holiday-vs-non-holiday-and-promotions)
    - [Correlation with Sales](#correlation-with-sales)
  - [🤖 Modeling and Results](#-modeling-and-results)
    - [Final selected model: Linear Regression](#final-selected-model-linear-regression)
  - [✅ Conclusions](#-conclusions)
  - [💡 Business Recommendations](#-business-recommendations)
  - [⚠️ Limitations](#️-limitations)
  - [🛠️ Technologies Used](#️-technologies-used)
  - [▶️ How to Reproduce the Project](#️-how-to-reproduce-the-project)
  - [🎓 Credits](#-credits)

---

## 🎯 Executive Summary

A retail store collected one year (365 days) of daily sales data and wants to **forecast next month's sales** in order to make informed decisions about inventory, promotions, and staffing.

A set of three regression models —**Linear Regression, Decision Tree, and Random Forest**— was built and compared to predict total sales based on the day of the week, promotions, and holidays.

**Key result:** all three models achieved virtually identical performance (**R² ≈ 0.992**), demonstrating that the relationship between the variables and sales is essentially **linear and simple**. **Linear Regression** was selected as the final model for delivering the same performance with the highest interpretability and lowest computational cost. The final model's average error is just **≈ 67 sales units (MAE)** against an average ticket of ~3,000.

---

## ❓ Problem Statement

> **Can the store anticipate its daily sales for next month using historical data from the previous year?**

The store needs a model that predicts total sales to support three critical operational decisions:

- 📦 **Inventory management** — how much product to stock and when.
- 🏷️ **Promotions** — assess whether the current promotion strategy adds value.
- 👥 **Staff scheduling** — assign more employees on the highest-demand days.

---

## 📊 Dataset

**File:** `Ventas.csv` — 365 records (full year 2022), no missing values.

| Variable | Type | Description |
|---|---|---|
| `Fecha` | date | Record date (daily, 2022-01-01 to 2022-12-31) |
| `DíaDeLaSemana` | integer (1–7) | Day of the week (1 = Monday … 7 = Sunday) |
| `Promociones` | binary (0/1) | Whether there was a promotion that day |
| `Festivo` | binary (0/1) | Whether the day was a holiday |
| `Ventas` | integer | **Target variable** — total sales for the day |

**Descriptive statistics of the target variable (`Ventas`):**

| Metric | Value |
|---|---|
| Count | 365 |
| Mean | 2,997.22 |
| Standard deviation | 942.10 |
| Minimum | 1,305 |
| Median (50%) | 3,074 |
| Maximum | 4,404 |

- Days with a **promotion:** 73 (20% of the year).
- **Holiday** days: 52 (14.2% of the year).

---

## 🔬 Methodology

The project followed a standard data science workflow:

```
1. Data Preparation  →  2. EDA  →  3. Model Selection  →  4. Training & Evaluation  →  5. Conclusions
```

1. **Data preparation** — loading the CSV, checking for nulls (0 found), and converting the `Fecha` column from text to `datetime`.
2. **Exploratory Data Analysis (EDA)** — sales distribution, sales over time, sales by day of week, holiday vs. non-holiday, promotion impact, correlation matrix, and interaction analysis.
3. **Model selection** — comparing three regression algorithms via R² on a test set.
4. **Training & evaluation** — an **80/20** *train/test* split (`random_state=42`) and visualization of actual vs. predicted sales.
5. **Conclusions and business recommendations.**

**Predictor variables (X):** `DíaDeLaSemana`, `Promociones`, `Festivo`
**Target variable (y):** `Ventas`

---

## 📈 Exploratory Data Analysis (EDA)

### Average sales by day of the week

Sales follow a **consistent, increasing** pattern throughout the week:

| Day | Mon | Tue | Wed | Thu | Fri | Sat | Sun |
|---|---|---|---|---|---|---|---|
| **Avg. sales** | 1,502 | 2,106 | 2,525 | 3,070 | 3,481 | 4,072 | 4,205 |

### Holiday vs. Non-Holiday and Promotions

| Segment | Avg. sales |
|---|---|
| **Non-holiday** day | 2,796.5 |
| **Holiday** day | 4,205.2 |
| **No** promotion | 2,973.8 |
| **With** promotion | 3,090.7 |

### Correlation with Sales

| Variable | Correlation with `Ventas` |
|---|---|
| `DíaDeLaSemana` | **0.987** 🟢 |
| `Festivo` | 0.523 🟡 |
| `Promociones` | 0.050 🔴 |

**EDA finding:** the **day of the week** completely dominates the variation in sales. An important structural detail in the data is that **all holidays fall on Sunday** (the highest-selling day) and **promotions only occur on Tuesdays, Thursdays, and Saturdays** — which explains why the apparent "holiday" effect is confounded with the weekend effect.

---

## 🤖 Modeling and Results

Three models were trained and evaluated on the same test set (20%):

| Model | R² (test) | MAE | RMSE |
|---|---|---|---|
| **Linear Regression** ✅ | **0.99242** | **67.43** | 86.24 |
| Decision Tree | 0.99235 | 65.94 | 86.66 |
| Random Forest | 0.99226 | 66.19 | 87.17 |

> All three models achieved **nearly identical performance (~99.2% R²)**. This confirms that the relationship between the variables and sales is fundamentally **linear**, so a complex model provides no advantage.

### Final selected model: Linear Regression

**Linear Regression** was chosen for matching the performance of more complex models while being the most **interpretable, lightweight, and easy to implement** option. The learned equation is:

```
Sales ≈ 1,028.67 + 492.95 · DayOfWeek + 181.07 · Promotion − 278.27 · Holiday
```

- Each day forward in the week adds **≈ 493 sales** — confirming the day of the week is the main driver.
- Once the day of the week is controlled for, **promotions add ≈ +181**, and the holiday variable turns negative, revealing that its raw effect came from coinciding with Sunday.

The **Actual vs. Predicted Sales** plot shows the points aligned very close to the ideal diagonal, visually confirming the model's high accuracy.

---

## ✅ Conclusions

1. **The day of the week is the strongest predictor of sales.** Sales are lowest at the start of the week (Monday ~1,500) and increase progressively toward the weekend (Saturday/Sunday ~4,100+). This single factor explains most of the variation.

2. **Promotions have a low impact.** Although days with promotions show slightly different sales, the effect is marginal compared to the day of the week. Only 20% of days had promotions, and sales during those promotions were practically the same.

3. **Holidays alter sales, but in a confounded way.** In raw terms, holidays sell more, but this is because **they all fall on Sunday** (the highest-selling day), not because of an intrinsic holiday effect.

4. **The data structure confirms the analysis.** There are no promotions on holidays; promotions only happen on Tuesdays, Thursdays, and Saturdays; and holidays only fall on Sundays.

5. **The three models perform almost identically (~99.2% R²).** The relationship is essentially linear and simple: no complex models are needed.

6. **Linear Regression is the most suitable model** because it delivers the same performance as more complex models while being easier to interpret and implement.

---

## 💡 Business Recommendations

| Area | Recommendation |
|---|---|
| 👥 **Staffing** | Schedule more employees on **Fridays, Saturdays, and Sundays**, the highest-selling days, or extend opening hours on those days. |
| 📦 **Inventory** | Stock more product toward the **weekend**. Avoid over-stocking on Mondays and Tuesdays to reduce storage costs. |
| 🏷️ **Promotions** | **Reassess the current strategy**, as it generates no measurable impact. Consider redirecting it to low-sales days (Monday–Wednesday) to stimulate them, or design special weekend promotions. |
| 🔮 **Predictive analytics** | Use the Linear Regression model to **forecast next month's sales** and plan inventory and staffing in advance. |

---

## ⚠️ Limitations

- **Only 3 predictor variables.** In reality, factors such as weather, competition, location, and annual seasonality also influence sales.
- **It is a practice dataset.** An R² of 0.99 is unusually high; in real-world scenarios models tend to have lower accuracy. It would be essential to **validate the model with new data** before making critical decisions.
- **Variables confounded by design** (holiday = Sunday, promotions only on certain days), which limits the ability to isolate each one's causal effect.

---

## 🛠️ Technologies Used

| Tool | Use |
|---|---|
| **Python 3** | Base language |
| **pandas** | Data loading and manipulation |
| **NumPy** | Numerical computation |
| **Matplotlib / seaborn** | Data visualization |
| **scikit-learn** | Modeling (`LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`), *train/test* split, and metrics |
| **Jupyter Notebook** | Interactive development environment |

---

## ▶️ How to Reproduce the Project

```bash
# 1. Install dependencies
pip install jupyter pandas numpy matplotlib seaborn scikit-learn

# 2. Open the notebook
jupyter notebook "07 Proyecto del dia 11.ipynb"

# 3. Run all cells (Cell → Run All)
```

**Project structure:**

```
Dia 11/
├── 07 Prediccion de ventas con ML.ipynb   # Notebook with the full analysis
├── Ventas.csv                      # Dataset (365 records)
├── README.md                       # Documentation (Spanish)
└── README_EN.md                    # This document (English)
```

---

## 🎓 Credits

This Machine Learning exercise was developed based on an exercise from a **Machine Learning course on Udemy taught by Federico Garay**.

---

<p align="center">
  <em>Educational project · Day 11 — Python for Data Science and Machine Learning</em>
</p>
