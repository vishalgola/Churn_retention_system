# Customer Churn Prediction System

## Quick Start

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Bugs Fixed from Original
1. **`customer_id` as feature** — caused identical prediction for every input
2. **Column name mismatches** — schema used `Geography`/`NumOfProducts` vs dataset's `country`/`products_number`
3. **Dashboard only sent 2 fields** — missing 8 required features
4. **Model never saved** — `train.py` was missing `pickle.dump` for the model
5. **No class imbalance handling** — added `class_weight='balanced'`

## Model
- Soft-voting ensemble: Gradient Boosting (w=3) + Random Forest (w=2) + Logistic Regression (w=1)
- ROC-AUC ≈ 0.87 on held-out test set
- 13 features (customer_id excluded)

## Structure
```
churn_app/
├── streamlit_app.py   # Main app (run this)
├── train.py           # Retrain model
├── requirements.txt
├── model_v2/          # Trained model artifacts
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── model_columns.pkl
└── data/
    └── Bank Customer Churn Prediction.csv
```
