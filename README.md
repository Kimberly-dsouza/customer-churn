# Bank Customer Churn Prediction
This project is a decision-support analytics project designed to help businesses proactively retain customers at risk of leaving. The model prioritizes high-risk customers for targeted retention actions, enabling marketing teams to allocate retention budgets more effectively. 

It is a churn-prediction project using the "Churn_Modelling.csv" dataset. Demonstrates data preparation, imbalance handling, model comparison, model persistence, and a demo UI. This README highlights business metrics and how the project can be used to drive actionable retention decisions.

<img width="215" height="307" alt="image" src="https://github.com/user-attachments/assets/677e13f0-a604-4200-b6bc-b390b72b6997" />

## What the project contains
- Notebook: `bank_customer_churn.ipynb` — full E2E analysis (preprocessing → SMOTE → scaling → model training → evaluation → save model → demo UI).
- Dataset: `Churn_Modelling.csv`
- Saved model: `churn_predict_model` (joblib)
- Requirements: `requirements.txt`

## Techniques used
- Data cleaning and categorical encoding (one-hot, drop_first)
- Class imbalance handling: SMOTE (imbalanced-learn)
- Feature scaling: StandardScaler for distance-based models
- Models compared: Logistic Regression, SVC, KNN, Decision Tree, Random Forest, Gradient Boosting
- Model persistence: joblib
- Simple Tkinter demo for manual prediction

## Business metrics and KPIs
We use multiple metrics because churn is imbalanced. Focus on business impact rather than raw accuracy.

- Churn rate (baseline) Use to set business targets and measure improvement. 
  - Churn rate = (# customers churned in period) / (total customers)

- Retention rate
  - Retention rate = 1 − churn rate

- Precision (positive predictive value)
  - Precision = TP / (TP + FP)
  - Proportion of customers flagged as "will churn" who actually churn. High precision => fewer wasted retention offers.

- Recall (sensitivity)
  - Recall = TP / (TP + FN)
  - Proportion of actual churners the model catches. High recall => fewer missed churners.

- F1 score
  - Harmonic mean of precision and recall. Useful when balancing false positives and false negatives.
 
## Cost metrics for ROI

- Cost_per_offer (C_offer): average cost to retain a customer (discount, incentive, outreach cost).
  - Lifetime_Value (LTV): expected net profit from a retained customer.
  - Cost_of_FN (C_fn): average lost profit when a churn is missed ≈ LTV.
  - Cost_of_FP (C_fp): cost of unnecessary retention ≈ C_offer.

- Business-weighted objective
  - Expected_savings = (TP * LTV) − ((FP * C_offer) + (Operations_costs))
  - Net_ROI = Expected_savings / (Total_cost_of_campaigns)
 
## Example ROI scenario
- Dataset baseline: 10,000 customers, true churn rate 20% → 2,000 churners
- Model performance on holdout:
  - Precision = 0.50, Recall = 0.60
- If model flags 2,400 customers as at-risk:
  - TP = 0.60 * 2,000 = 1,200
  - FP = 2,400 − 1,200 = 1,200
- Assume:
  - C_offer = $30
  - LTV = $400
- Savings from prevented churn (approx) = TP * LTV = 1,200 * $400 = $480,000
- Cost of offers = (TP + FP) * C_offer = 2,400 * $30 = $72,000
- Net benefit ≈ $480,000 − $72,000 = $408,000
- ROI = Net benefit / Cost_of_offers ≈ $408,000 / $72,000 ≈ 5.67x











