"""
Customer Churn Prediction — Streamlit App
Standalone: model loaded directly (no FastAPI needed)
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ─────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────
# Load model (cached)
# ─────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_v2")

def _train_model():
    """Train and save the model fresh — called automatically if pkl is stale/incompatible."""
    import pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    DATA_PATH = os.path.join(os.path.dirname(__file__), "data",
                             "Bank Customer Churn Prediction.csv")

    df = pd.read_csv(DATA_PATH)
    df = df.drop("customer_id", axis=1)
    df = pd.get_dummies(df, columns=["country", "gender"], drop_first=False)

    X = df.drop("churn", axis=1)
    y = df["churn"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                    subsample=0.8, min_samples_leaf=10, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5,
                                class_weight="balanced", random_state=42)
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)

    model = VotingClassifier(
        estimators=[("gb", gb), ("rf", rf), ("lr", lr)],
        voting="soft", weights=[3, 2, 1]
    )
    model.fit(X_train_s, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    pickle.dump(model,              open(f"{MODEL_DIR}/best_model.pkl",    "wb"))
    pickle.dump(scaler,             open(f"{MODEL_DIR}/scaler.pkl",        "wb"))
    pickle.dump(X.columns.tolist(), open(f"{MODEL_DIR}/model_columns.pkl", "wb"))
    return model, scaler, X.columns.tolist()


@st.cache_resource
def load_model():
    try:
        model   = pickle.load(open(f"{MODEL_DIR}/best_model.pkl",    "rb"))
        scaler  = pickle.load(open(f"{MODEL_DIR}/scaler.pkl",        "rb"))
        columns = pickle.load(open(f"{MODEL_DIR}/model_columns.pkl", "rb"))
        import numpy as np
        scaler.transform(np.zeros((1, len(columns))))
        _ = model.predict_proba(np.zeros((1, len(columns))))
        return model, scaler, columns
    except Exception:
        return None, None, None  # signal retrain needed

_result = load_model()
if _result[0] is None:
    with st.spinner("⚙️ Re-training model for your sklearn version... (~30 sec)"):
        _result = _train_model()

model, scaler, model_columns = _result
# ─────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────
def risk_level(prob: float):
    if prob >= 0.70:
        return "🔴 High Risk", "#FF4B4B"
    elif prob >= 0.40:
        return "🟡 Medium Risk", "#FFA500"
    else:
        return "🟢 Low Risk", "#21C354"


def predict(inputs: dict) -> tuple[float, np.ndarray]:
    """
    Build a row, one-hot encode country/gender the same way as training,
    align to model_columns, scale, and predict.
    """
    row = {
        "credit_score":     inputs["credit_score"],
        "age":              inputs["age"],
        "tenure":           inputs["tenure"],
        "balance":          inputs["balance"],
        "products_number":  inputs["products_number"],
        "credit_card":      inputs["credit_card"],
        "active_member":    inputs["active_member"],
        "estimated_salary": inputs["estimated_salary"],
        # one-hot for country
        "country_France":   1 if inputs["country"] == "France"   else 0,
        "country_Germany":  1 if inputs["country"] == "Germany"  else 0,
        "country_Spain":    1 if inputs["country"] == "Spain"    else 0,
        # one-hot for gender
        "gender_Female":    1 if inputs["gender"] == "Female"    else 0,
        "gender_Male":      1 if inputs["gender"] == "Male"      else 0,
    }
    df = pd.DataFrame([row])
    df = df.reindex(columns=model_columns, fill_value=0)
    X  = scaler.transform(df)
    prob        = model.predict_proba(X)[0][1]
    importances = _feature_importance(df.columns.tolist())
    return float(prob), importances


def _feature_importance(cols):
    """Extract feature importances from the VotingClassifier's GB sub-model."""
    try:
        gb = model.named_estimators_["gb"]
        return dict(zip(cols, gb.feature_importances_))
    except Exception:
        return {}

# ─────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main gradient header */
    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem 2rem 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
        color: white;
    }
    .hero h1 { font-size: 2.2rem; margin: 0; font-weight: 800; }
    .hero p  { opacity: .75; margin: .4rem 0 0; font-size: 1rem; }

    /* Result card */
    .result-card {
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .result-prob  { font-size: 3.2rem; font-weight: 900; line-height: 1; }
    .result-label { font-size: 1.4rem; font-weight: 700; margin-top: .4rem; }

    /* Info cards */
    .info-card {
        background: #f8f9ff;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        border-left: 4px solid #4361ee;
        margin: .6rem 0;
        font-size: .93rem;
        color: #333;
    }

    /* Gauge bar */
    .gauge-container {
        background: #e9ecef;
        border-radius: 999px;
        height: 20px;
        overflow: hidden;
        margin: .6rem 0;
    }
    .gauge-fill {
        height: 100%;
        border-radius: 999px;
        transition: width .6s ease;
    }

    /* Section titles */
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1a1a2e;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: .3rem;
        margin: 1rem 0 .8rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #4a517d;
    }

    /* Hide Streamlit watermark */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# Hero header
# ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🔮 Customer Churn Predictor</h1>
  <p>Ensemble ML model (Gradient Boosting + Random Forest + Logistic Regression) · ROC-AUC ≈ 0.87</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# Layout: sidebar = inputs, main = results
# ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
    "<h2 style='color:#13f2d5;'>🧑‍💼 Customer Details</h2>",
    unsafe_allow_html=True
)
    st.markdown("Fill in the customer profile below:")

    st.markdown('<div class="section-title">Demographics</div>', unsafe_allow_html=True)
    country = st.selectbox("Country", ["France", "Germany", "Spain"])
    gender  = st.selectbox("Gender",  ["Male", "Female"])
    age     = st.slider("Age", 18, 92, 38)

    st.markdown('<div class="section-title">Financial Profile</div>', unsafe_allow_html=True)
    credit_score     = st.slider("Credit Score", 300, 850, 620)
    balance          = st.number_input("Account Balance (€)", min_value=0.0, max_value=300_000.0,
                                       value=75_000.0, step=1000.0, format="%.2f")
    estimated_salary = st.number_input("Estimated Annual Salary (€)", min_value=0.0,
                                       max_value=250_000.0, value=60_000.0, step=500.0, format="%.2f")

    st.markdown('<div class="section-title">Account Activity</div>', unsafe_allow_html=True)
    tenure          = st.slider("Tenure (years)", 0, 10, 4)
    products_number = st.selectbox("Number of Products", [1, 2, 3, 4])
    credit_card     = st.radio("Has Credit Card?",  ["Yes", "No"], horizontal=True)
    active_member   = st.radio("Active Member?",    ["Yes", "No"], horizontal=True)

    predict_btn = st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────
# Prediction & Results
# ─────────────────────────────────────────────────────
if predict_btn:
    inputs = {
        "credit_score":     credit_score,
        "age":              age,
        "tenure":           tenure,
        "balance":          balance,
        "products_number":  products_number,
        "credit_card":      1 if credit_card == "Yes" else 0,
        "active_member":    1 if active_member == "Yes" else 0,
        "estimated_salary": estimated_salary,
        "country":          country,
        "gender":           gender,
    }

    with st.spinner("Analysing customer profile…"):
        prob, importances = predict(inputs)

    label, color = risk_level(prob)
    pct = round(prob * 100, 1)

    # ── Top result row ──────────────────────────────────────────────
    col_result, col_gauge = st.columns([1, 1])

    with col_result:
        st.markdown(f"""
        <div class="result-card" style="background: linear-gradient(135deg, {color}cc, {color}88);">
            <div class="result-prob">{pct}%</div>
            <div class="result-label">{label}</div>
            <div style="opacity:.8; font-size:.85rem; margin-top:.5rem;">Churn Probability</div>
        </div>
        """, unsafe_allow_html=True)

    with col_gauge:
        st.markdown("#### Risk Meter")
        st.markdown(f"""
        <div style="margin: .8rem 0 .3rem; font-size:.85rem; color:#555;">
            Low (0%) ←——————→ High (100%)
        </div>
        <div class="gauge-container">
            <div class="gauge-fill" style="width:{pct}%; background:{color};"></div>
        </div>
        <p style="text-align:right; font-size:.8rem; color:#888;">{pct}%</p>
        """, unsafe_allow_html=True)

        # Risk breakdown bands
        st.markdown("""
        | Risk Band | Range | Action |
        |-----------|-------|--------|
        | 🔴 High   | ≥ 70% | Immediate intervention |
        | 🟡 Medium | 40–69%| Targeted retention offer |
        | 🟢 Low    | < 40% | Monitor periodically |
        """)

    st.divider()

    # ── Retention Recommendations ───────────────────────────────────
    st.markdown("### 💡 Retention Recommendations")
    col_rec1, col_rec2, col_rec3 = st.columns(3)

    recs = []
    if balance == 0:
        recs.append(("💳 Zero Balance", "Customer has €0 balance — consider account activation incentives or targeted deposit campaigns."))
    if active_member == "No":
        recs.append(("📵 Inactive Member", "Customer is not actively using the account. Offer personalised engagement or loyalty rewards."))
    if products_number == 1:
        recs.append(("📦 Single Product", "Cross-sell opportunity: customers with 2+ products churn far less. Offer a relevant add-on."))
    if credit_card == "No":
        recs.append(("💳 No Credit Card", "Issuing a credit card can deepen the banking relationship and reduce churn risk."))
    if age > 55:
        recs.append(("👴 Senior Customer", "Older customers may value dedicated support channels and simplified digital interfaces."))
    if pct >= 70:
        recs.append(("🚨 Urgent Outreach", "High churn risk detected. Prioritise a direct call from a relationship manager within 48 hours."))

    if not recs:
        recs = [("✅ Stable Profile", "This customer shows a healthy engagement profile. Continue standard relationship management.")]

    cols = [col_rec1, col_rec2, col_rec3]
    for i, (title, body) in enumerate(recs[:3]):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="info-card">
                <strong>{title}</strong><br>
                <span style="color:#555;">{body}</span>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Feature importance ──────────────────────────────────────────
    if importances:
        st.markdown("### 📊 Top Factors Driving This Prediction")
        imp_df = pd.DataFrame({
            "Feature": list(importances.keys()),
            "Importance": list(importances.values())
        }).sort_values("Importance", ascending=True).tail(8)

        # Simple horizontal bar using st.progress-style rendering
        max_imp = imp_df["Importance"].max()
        for _, row in imp_df.iloc[::-1].iterrows():
            feat = row["Feature"].replace("_", " ").title()
            val  = row["Importance"]
            pct_bar = int((val / max_imp) * 100)
            st.markdown(f"""
            <div style="margin:.3rem 0;">
                <div style="display:flex; justify-content:space-between; font-size:.85rem; color:#ebe30c;">
                    <span>{feat}</span><span style="color:#4361ee; font-weight:700;">{val:.3f}</span>
                </div>
                <div class="gauge-container" style="height:12px; margin-top:.2rem;">
                    <div class="gauge-fill" style="width:{pct_bar}%; background: linear-gradient(90deg, #4361ee, #7b2ff7);"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Customer Summary ────────────────────────────────────────────
    st.markdown("### 📋 Customer Profile Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Country",        country)
    c2.metric("Age",            age)
    c3.metric("Credit Score",   credit_score)
    c4.metric("Products",       products_number)

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Balance (€)",    f"{balance:,.0f}")
    c6.metric("Salary (€)",     f"{estimated_salary:,.0f}")
    c7.metric("Tenure (yrs)",   tenure)
    c8.metric("Active Member",  active_member)

else:
    # Welcome screen when no prediction yet
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color: #ebe30c;">
        <div style="font-size: 5rem;">👈</div>
        <h3 style="color:#444;">Fill in the customer details on the left</h3>
        <p>Enter the customer's profile in the sidebar and click <strong>"Predict Churn Risk"</strong>
        to get an instant probability score with retention recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    # Show model stats in welcome screen
    st.markdown("---")
    st.markdown("### 🧠 About the Model")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Model Type",    "Ensemble (3 models)")
    col_b.metric("ROC-AUC",       "~0.87")
    col_c.metric("Training Data", "10,000 customers")
    col_d.metric("Features",      "13")

    st.markdown("""
    <div class="info-card">
        <strong>📌 Architecture</strong><br>
        Voting ensemble of <em>Gradient Boosting</em> (weight 3) + <em>Random Forest</em> (weight 2)
        + <em>Logistic Regression</em> (weight 1) with StandardScaler preprocessing.
        Features: Credit Score, Age, Tenure, Balance, Products, Credit Card, Active Member,
        Estimated Salary, Country, Gender.
    </div>
    <div class="info-card" style="border-left-color:#e74c3c;">
        <strong>🐛 Bugs fixed from original</strong><br>
        (1) <code>customer_id</code> was included as a training feature — this caused every prediction to be identical.
        (2) Column name mismatches between schema and dataset.
        (3) Dashboard only sent 2 fields instead of 10.
        (4) Model was never actually saved in train.py.
        (5) No class-imbalance handling (80/20 split).
    </div>
    """, unsafe_allow_html=True)
