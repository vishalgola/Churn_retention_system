# 🚀 Churn Retention System  
### Customer Churn Prediction using Ensemble Machine Learning  

---

## 📌 Overview  
Customer churn is one of the biggest hidden revenue leaks in any business. Most companies react *after* customers leave — this system focuses on predicting churn **before it happens**.

This project is a **Machine Learning-powered churn prediction system** built using an ensemble of models to improve prediction reliability and decision-making.

It is deployed using **Streamlit**, enabling real-time predictions through an interactive UI.

---

## 🎯 Problem Statement  
Traditional single-model approaches often fail to capture complex customer behavior patterns.

This system solves that by:
- Predicting churn probability  
- Classifying customers into actionable risk levels  
- Enabling quick, data-driven retention decisions  

---

## ⚙️ Tech Stack  

- **Machine Learning:** Scikit-learn  
- **Models Used:**  
  - Gradient Boosting  
  - Random Forest  
  - Linear Regression *(used for probability estimation in ensemble)*  
- **Frontend / Deployment:** Streamlit  
- **Data Processing:** Pandas, NumPy  
- **Model Storage:** Pickle  

---

## 🧠 Ensemble Strategy  

Instead of relying on a single model, this system combines multiple models:

- **Random Forest** → Captures complex non-linear relationships  
- **Gradient Boosting** → Improves prediction accuracy by correcting errors  
- **Linear Regression** → Provides smooth probability estimation  

### 🔥 Final Prediction Logic  
- Predictions from all models are combined using **Averaging (Ensemble Voting)**  
- Output is converted into **churn probability + risk category + Retention Recommendations**

---

## 🏗️ Project Structure  

```
Churn_retention_system/
│
├── data/
│   └── Bank Customer Churn.csv
│
├── model_v2/
│   ├── best_model.pkl
│   ├── model_columns.pkl
│   ├── scaler.pkl
│
├── streamlit_app.py
├── train.py
├── requirements.txt
└── README.md
```

---

## 🔥 Key Features  

- ✅ Ensemble-based prediction (more robust than single models)  
- ✅ Real-time churn prediction  
- ✅ Risk segmentation (Low / Medium / High)  
- ✅ Clean and interactive UI  
- ✅ Scalable architecture for deployment  

---

## 🚀 Live Demo  

👉 **Try the App:**  
[Have a Look](https://churn-retention-system-app1.streamlit.app/)  

---

## 🖥️ Run Locally  

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 📊 Model Performance  

| Metric       |  Score    |
|--------------|-----------|
| ROC-AUC      | **86.7%** |
| Accuracy     | **~85%**  |
| Precision    | **~85%**  |
| Recall       | **~85%**  |

---

## 📸 App Preview  

<img width="957" height="450" alt="image" src="https://github.com/user-attachments/assets/76bcfec1-e9ac-4775-996a-ccefb6ca2c91" />


---

## 🧪 How It Works  

1. User inputs customer details  
2. Data is preprocessed using:
   - Saved feature columns  
   - Standard scaler  
3. Ensemble model predicts churn probability  
4. Output is classified into:
   - 🟢 Low Risk  
   - 🟡 Medium Risk  
   - 🔴 High Risk  

---

## ⚠️ Limitations  

- Dataset-dependent performance  
- No explainability (yet)  
- Basic ensemble (not stacking)  

---

## 💡 Future Improvements  

- 🔥 Implement **Stacking Ensemble (major upgrade)**  
- 📊 Add **SHAP / Explainable AI**  
- ⚙️ Deploy using **FastAPI + Docker**  
- ⚖️ Handle class imbalance more effectively  
- 📈 Add business insights dashboard  

---

## 📬 Contact  

**Vishal Prajapati**  

- 🔗 LinkedIn: https://www.linkedin.com/in/vishal-prajapati93/  
- 📧 Email: vishalprajapati935498@gmail.com 

---

## ⭐ Final Note  

This project is not just about prediction — it's about **turning data into retention strategy**.

If you're building ML projects and not solving a real business problem, you're just playing with models.  
This one actually solves something.
