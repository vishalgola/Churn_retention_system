import pandas as pd
import pickle
import os
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

os.makedirs("model_v2", exist_ok=True)

df = pd.read_csv("data/Bank Customer Churn Prediction.csv")
df = df.drop("customer_id", axis=1)  # never include IDs as features
df = pd.get_dummies(df, columns=["country", "gender"], drop_first=False)

X = df.drop("churn", axis=1)
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                subsample=0.8, min_samples_leaf=10, random_state=42)
rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5,
                            class_weight="balanced", random_state=42)
lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)

model = VotingClassifier(estimators=[("gb", gb), ("rf", rf), ("lr", lr)],
                         voting="soft", weights=[3, 2, 1])
model.fit(X_train_s, y_train)

probs = model.predict_proba(X_test_s)[:, 1]
preds = (probs >= 0.45).astype(int)
print("ROC-AUC:", round(roc_auc_score(y_test, probs), 4))
print(classification_report(y_test, preds))

pickle.dump(model,            open("model_v2/best_model.pkl",    "wb"))
pickle.dump(scaler,           open("model_v2/scaler.pkl",        "wb"))
pickle.dump(X.columns.tolist(), open("model_v2/model_columns.pkl", "wb"))
print("Saved to model_v2/")
