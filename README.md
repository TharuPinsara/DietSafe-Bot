# DIETSAFE BOT — Predicting Food Healthiness

An AI/ML project that classifies recipes from the Epicurious Recipe Dataset as
**healthy** or **unhealthy** based on nutritional features (calories, fat, protein,
sodium, fiber, etc.), supporting home cooks, restaurants, and healthcare providers
in making safer, more informed dietary choices — including allergy-aware and
diet-related-disease-aware recommendations.

## Team

| Admission No. | Name |
|---|---|
| IT24102205 | Wijekoon S.P.A.S.A |
| IT24102239 | Vithanage R.A.I |
| IT24102307 | Dewmith H. L. T. P |
| IT24102241 | Rupasingha R.S.M.P.S |
| IT24102242 | Venura Kithpura W.K |
| IT24102286 | Hisham M |

## Dataset

- **Source:** [Epicurious – Recipes with Rating and Nutrition](https://www.kaggle.com/datasets/hugodarwood/epirecipes) (Kaggle)
- **Raw size:** 20,053 rows × 680 columns
- **Target:** Binary `healthy` label, derived from an engineered `health_score`
  (weighted combination of fiber, protein, calories, fat, and sodium)
- **Final processed size:** 10,619 rows × 11 features (after cleaning, encoding, outlier removal, and scaling)

## Pipeline

```
Raw data → Feature selection → Missing-value handling → Categorical encoding
→ Outlier removal (IQR) → Feature engineering (health_score) → Correlation-based
feature selection → Min-Max scaling → Train/test split (80/20) → SMOTE →
Model training + GridSearchCV tuning → Evaluation
```

## Notebooks

| Notebook | Stage |
|---|---|
| `AIML.ipynb` | EDA / text preprocessing & scaling |
| `Logistic_Regression.ipynb` | Baseline linear model + feature engineering |
| `Random_Forest.ipynb` | Ensemble bagging model |
| `KNN_model.ipynb` | Instance-based classifier |
| `SVM.ipynb` | Margin-based classifier (best-performing model) |
| `Decision_Tree.ipynb` | Baseline tree model |

Each notebook covers one stage of preprocessing and/or one classification model,
all trained on the same cleaned, engineered dataset with `StandardScaler`
normalization and `GridSearchCV` hyperparameter tuning (3–5 fold CV).

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **SVM** (best) | **0.99** | 0.99 | 0.99 | 0.99 |
| Random Forest | 0.98 | 0.98 | 0.98 | 0.98 |
| Decision Tree | 0.97 | 0.91 | 0.93 | 0.98 |
| Gradient Boosting | 0.96 | 0.95 | 0.97 | 0.96 |
| KNN | 0.92 | 0.92 | 0.92 | 0.92 |
| Logistic Regression | 0.89 | 0.99 | 0.99 | 0.90 |

**Best model:** SVM, chosen for its highest accuracy/precision/recall/F1 and its
ability to handle high-dimensional, non-linear nutritional data via kernel tricks.

## Ethical Considerations

- **Health safety first:** models prioritized for high recall to minimize false
  negatives (unsafe recipes misclassified as healthy).
- **Bias mitigation:** SMOTE for class balance; fairness audits across fat levels.
- **Dataset limitation:** Western-centric (Epicurious), likely underrepresenting
  global cuisines — flagged for future work.
- **Transparency:** the bot is not a substitute for professional medical or
  dietary advice.

## References

- Scikit-learn Documentation — https://scikit-learn.org/stable/
- Kaggle: Epicurious Recipes Dataset — https://www.kaggle.com/datasets/hugodarwood/epirecipes
- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR.
