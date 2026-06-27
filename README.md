# Adult Income Classifier

Predicting whether a person earns more or less than $50K/year using the 
UCI Adult Income (Census) dataset. Built to practice the full ML workflow 
end to end — not just calling a model and hoping for the best.

## What This Project Covers

- **EDA** — distributions, missing values encoded as `?` strings, class 
  imbalance analysis, correlation heatmap
- **Preprocessing Pipeline** — sklearn `Pipeline` + `ColumnTransformer` 
  handling numerical and categorical columns separately. No data leakage.
- **Models** — Decision Tree, Random Forest, XGBoost
- **Hyperparameter Tuning** — GridSearchCV with 5-fold cross-validation, 
  optimizing for ROC-AUC
- **GPU Acceleration** — XGBoost trained with CUDA on RTX 3060

## Results

| Model | Default ROC-AUC | Tuned ROC-AUC | Improvement |
|---|---|---|---|
| Decision Tree | 0.772 | 0.902 | +0.130 |
| Random Forest | 0.888 | 0.910 | +0.022 |
| XGBoost | 0.925 | 0.926 | +0.001 |

XGBoost wins overall — best accuracy (87%), best F1 for class 1 (0.70), 
best ROC-AUC (0.926).

## Key Lessons

- Always split before preprocessing — pipelines enforce this automatically
- Accuracy is misleading on imbalanced data — ROC-AUC tells the real story
- Single Decision Trees are the most sensitive to hyperparameter tuning — 
  biggest improvement by far
- GPU acceleration made XGBoost training go from ~5 mins (CPU estimate) 
  to 43 seconds

## Tech used

- Python, Jupyter Notebook
- pandas, numpy, matplotlib, seaborn
- scikit-learn, XGBoost

## Dataset

[UCI Adult Income Dataset](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset)

## How to Run

```bash
git clone https://github.com/AMokhtar51/adult-income-classifier
cd adult-income-classifier
pip install -r requirements.txt
jupyter notebook
```