# 🚀 Churn Retention System  
**Customer Churn Prediction using Ensemble Machine Learning & Streamlit**

---

## 📌 Overview  
This project is a Machine Learning-based churn prediction system that uses an ensemble of multiple models to improve prediction accuracy and robustness.

It is deployed using a Streamlit web app, allowing users to input customer data and get real-time churn predictions.

---

## 🎯 Problem Statement  
Customer churn leads to major revenue loss.  
Single models often fail to capture complex patterns in customer behavior.

This system helps by:
- Predicting churn probability  
- Classifying customers into risk levels  
- Providing a simple interface for quick decision-making  

---

## ⚙️ Tech Stack  

- Machine Learning: Scikit-learn  
- Models Used:
  - Gradient Boosting  
  - Random Forest  
  - Logistic Regression  
- Frontend / Deployment: Streamlit  
- Data Processing: Pandas, NumPy  
- Model Storage: Pickle  

---

## 🧠 Ensemble Strategy  

This project uses an ensemble learning approach:

- Random Forest captures non-linear patterns  
- Gradient Boosting improves performance on complex relationships  
- Logistic Regression provides baseline comparison  

Final prediction is made using:
- Voting / Averaging mechanism (update if stacking is used)  

---

## 🏗️ Project Structure  

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

---

## 🔥 Key Features  

- Ensemble-based Churn Prediction  
- Improved Accuracy vs Single Models  
- Risk Classification (Low / Medium / High)  
- Interactive UI using Streamlit  
- Real-time predictions  

---

## 🚀 How It Works  

1. User inputs customer details  
2. Data is preprocessed using scaler and saved columns  
3. Ensemble model predicts churn probability  
4. Output shows prediction with risk level  

---

## 🖥️ Run Locally  

    pip install -r requirements.txt
    streamlit run streamlit_app.py

---

## 📊 Model Performance  

(Add your real metrics here)

- Accuracy: XX%  
- ROC-AUC: XX  

---

## 📸 App Preview  

(Add screenshot here)

![App Screenshot](your-image-link)

---

## 💡 Future Improvements  

- Implement Stacking Ensemble  
- Deploy using FastAPI + Docker  
- Add model explainability (SHAP)  
- Improve class imbalance handling  

---

## 📬 Contact  

Vishal Prajapati  
[(Add LinkedIn and Email)](https://www.linkedin.com/in/vishal-prajapati93/)

---

## ⚠️ Notes  

- Update performance metrics with actual values  
- Add screenshots for better presentation  
- Clearly define ensemble method (Voting / Stacking)
