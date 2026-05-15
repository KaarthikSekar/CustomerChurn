"""
Customer Churn Prediction Dashboard
Logistic Regression model trained on CustomerChurn.csv — no CSV upload required.
Run: streamlit run app.py
"""

import math
import streamlit as st

# ─────────────────────────────────────────────────────────────
# Embedded model (Logistic Regression, AUC ≈ 0.855)
# Trained on Telco CustomerChurn.csv with SMOTE upsampling
# ─────────────────────────────────────────────────────────────
MODEL = {
    "intercept": -0.12342748045901206,
    "coef": {
        "gender": -0.007175291650309579,
        "SeniorCitizen": 0.09209660995359532,
        "Partner": 0.04184248472887192,
        "Dependents": -0.10223836207858923,
        "tenure": -0.8417047690655524,
        "PhoneService": -0.27243012022704705,
        "MultipleLines": 0.10898665314067628,
        "InternetService": 0.14419903871026074,
        "OnlineSecurity": -0.18649604262395045,
        "OnlineBackup": -0.06779985905358811,
        "DeviceProtection": -0.04923485800209184,
        "TechSupport": -0.17851530604099375,
        "StreamingTV": 0.06729377511794196,
        "StreamingMovies": 0.09949938327071658,
        "Contract": -0.6976763996643062,
        "PaperlessBilling": 0.17038604866947746,
        "PaymentMethod": 0.05748155113235918,
        "MonthlyCharges": 0.4665294469261137,
        "TotalCharges": 0.5099303490531717,
        "charge_per_tenure": 0.48374834397454525,
        "has_streaming": -0.11919043220795397,
        "has_security": -0.015235870646141195,
    },
    "scaler_mean": {
        "gender": 0.50748973181928,
        "SeniorCitizen": 0.19014254650881857,
        "Partner": 0.44491423049045664,
        "Dependents": 0.25718772650398647,
        "tenure": 27.624667794153176,
        "PhoneService": 0.9021502778448901,
        "MultipleLines": 0.9596520898767819,
        "InternetService": 0.8512925827494564,
        "OnlineSecurity": 0.6681565595554482,
        "OnlineBackup": 0.8140855279052911,
        "DeviceProtection": 0.8107030683740034,
        "TechSupport": 0.660908431988403,
        "StreamingTV": 0.9588064749939599,
        "StreamingMovies": 0.9539743899492631,
        "Contract": 0.5118386083595071,
        "PaperlessBilling": 0.6478618023677216,
        "PaymentMethod": 1.6372312152693886,
        "MonthlyCharges": 67.59266731094469,
        "TotalCharges": 2025.1679964967382,
        "charge_per_tenure": 7.7671763031710395,
        "has_streaming": 0.678183136023194,
        "has_security": 0.6106547475235564,
    },
    "scaler_scale": {
        "gender": 0.49994390077015166,
        "SeniorCitizen": 0.39241350450253404,
        "Partner": 0.4969562938504165,
        "Dependents": 0.43708374465278055,
        "tenure": 24.03538615730784,
        "PhoneService": 0.297111349546391,
        "MultipleLines": 0.9489585470364756,
        "InternetService": 0.6773473024439681,
        "OnlineSecurity": 0.8507508493222156,
        "OnlineBackup": 0.8940541849654828,
        "DeviceProtection": 0.8933441376454712,
        "TechSupport": 0.8478879555059056,
        "StreamingTV": 0.9122500589674549,
        "StreamingMovies": 0.9120190347838171,
        "Contract": 0.7697107749155602,
        "PaperlessBilling": 0.477636773501129,
        "PaymentMethod": 1.0169858052029501,
        "MonthlyCharges": 28.783704289568984,
        "TotalCharges": 2179.800536109581,
        "charge_per_tenure": 10.388886426234352,
        "has_streaming": 0.4671731692177323,
        "has_security": 0.4876018117793431,
    },
    "encoders": {
        "gender": ["Female", "Male"],
        "Partner": ["No", "Yes"],
        "Dependents": ["No", "Yes"],
        "PhoneService": ["No", "Yes"],
        "MultipleLines": ["No", "No phone service", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["No", "No internet service", "Yes"],
        "OnlineBackup": ["No", "No internet service", "Yes"],
        "DeviceProtection": ["No", "No internet service", "Yes"],
        "TechSupport": ["No", "No internet service", "Yes"],
        "StreamingTV": ["No", "No internet service", "Yes"],
        "StreamingMovies": ["No", "No internet service", "Yes"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["No", "Yes"],
        "PaymentMethod": [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check",
        ],
    },
    "feature_order": [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "charge_per_tenure",
        "has_streaming",
        "has_security",
    ],
}

# ─────────────────────────────────────────────────────────────
# Dataset-level insight figures (pre-computed from CSV)
# ─────────────────────────────────────────────────────────────
STATS = {
    "churn_by_contract": {
        "Month-to-month": 0.427,
        "One year": 0.113,
        "Two year": 0.028,
    },
    "churn_by_internet": {
        "Fiber optic": 0.419,
        "DSL": 0.190,
        "No": 0.074,
    },
    "churn_by_payment": {
        "Electronic check": 0.453,
        "Mailed check": 0.191,
        "Bank transfer (automatic)": 0.167,
        "Credit card (automatic)": 0.152,
    },
}


# ─────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────
def encode(col, value):
    return MODEL["encoders"][col].index(value)


def predict(inputs: dict):
    raw = dict(inputs)
    for col in MODEL["encoders"]:
        raw[col] = encode(col, raw[col])

    raw["charge_per_tenure"] = raw["MonthlyCharges"] / (raw["tenure"] + 1)
    raw["has_streaming"] = int(
        encode("StreamingTV", inputs["StreamingTV"]) == 2
        or encode("StreamingMovies", inputs["StreamingMovies"]) == 2
    )
    raw["has_security"] = int(
        encode("OnlineSecurity", inputs["OnlineSecurity"]) == 2
        or encode("DeviceProtection", inputs["DeviceProtection"]) == 2
    )

    logit = MODEL["intercept"]
    contributions = {}
    for feat in MODEL["feature_order"]:
        z = (raw[feat] - MODEL["scaler_mean"][feat]) / MODEL["scaler_scale"][feat]
        contrib = z * MODEL["coef"][feat]
        contributions[feat] = contrib
        logit += contrib

    prob = 1.0 / (1.0 + math.exp(-logit))
    return prob, contributions


# ─────────────────────────────────────────────────────────────
# Page config & CSS
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
 
html, body, [class*="css"]  { font-family: 'DM Sans', sans-serif; }
h1, h2, h3                  { font-family: 'Space Mono', monospace; }
 
.metric-card {
    background: #0f172a; border: 1px solid #1e293b;
    border-radius: 12px; padding: 18px 22px; text-align: center;
}
.metric-card .val {
    font-family: 'Space Mono', monospace;
    font-size: 2rem; font-weight: 700; color: #38bdf8; line-height: 1;
}
.metric-card .lbl {
    font-size: 0.75rem; color: #64748b; margin-top: 6px;
    text-transform: uppercase; letter-spacing: 0.08em;
}
 
.risk-high { background: linear-gradient(135deg,#7f1d1d,#450a0a); border: 1px solid #ef4444; border-radius:16px; padding:28px 32px; text-align:center; }
.risk-low  { background: linear-gradient(135deg,#052e16,#022c22); border: 1px solid #22c55e; border-radius:16px; padding:28px 32px; text-align:center; }
.risk-med  { background: linear-gradient(135deg,#422006,#1c1003); border: 1px solid #f59e0b; border-radius:16px; padding:28px 32px; text-align:center; }
 
.risk-title { font-family:'Space Mono',monospace; font-size:1rem; color:#cbd5e1; }
.risk-pct   { font-family:'Space Mono',monospace; font-size:3.5rem; font-weight:700; line-height:1.1; }
.risk-sub   { font-size:0.8rem; color:#94a3b8; margin-top:8px; }
 
.bar-wrap { margin: 5px 0; }
.bar-label { font-size:0.72rem; color:#94a3b8; margin-bottom:2px; display:flex; justify-content:space-between; }
.bar-bg    { background:#1e293b; border-radius:99px; height:8px; overflow:hidden; }
.bar-fill  { height:8px; border-radius:99px; }
 
section[data-testid="stSidebar"] { background:#0a0f1e; border-right:1px solid #1e293b; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# Sidebar — inputs
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👤 Customer Profile")
    st.markdown("---")

    st.markdown("**Demographics**")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox(
        "Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No"
    )
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

    st.markdown("---")
    st.markdown("**Account**")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    monthly_charges = st.number_input(
        "Monthly Charges ($)", 18.0, 119.0, 65.0, step=0.5
    )
    total_charges = st.number_input(
        "Total Charges ($)",
        0.0,
        9000.0,
        float(round(tenure * monthly_charges, 2)),
        step=10.0,
    )

    st.markdown("---")
    st.markdown("**Services**")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    no_inet = "No internet service"

    def inet_opts():
        return ["No", "Yes", no_inet] if internet != "No" else [no_inet]

    def inet_default(val="No"):
        opts = inet_opts()
        return opts.index(val) if val in opts else 0

    online_sec = st.selectbox("Online Security", inet_opts(), index=0)
    online_bkp = st.selectbox("Online Backup", inet_opts(), index=0)
    device_prot = st.selectbox("Device Protection", inet_opts(), index=0)
    tech_sup = st.selectbox("Tech Support", inet_opts(), index=0)
    stream_tv = st.selectbox("Streaming TV", inet_opts(), index=0)
    stream_mov = st.selectbox("Streaming Movies", inet_opts(), index=0)

# ─────────────────────────────────────────────────────────────
# Run inference always (live updates)
# ─────────────────────────────────────────────────────────────
inputs = {
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet,
    "OnlineSecurity": online_sec,
    "OnlineBackup": online_bkp,
    "DeviceProtection": device_prot,
    "TechSupport": tech_sup,
    "StreamingTV": stream_tv,
    "StreamingMovies": stream_mov,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
}

prob, contribs = predict(inputs)
pct = prob * 100

if pct >= 60:
    cls, emoji, label, color = "risk-high", "🔴", "HIGH RISK", "#ef4444"
elif pct >= 35:
    cls, emoji, label, color = "risk-med", "🟡", "MEDIUM RISK", "#f59e0b"
else:
    cls, emoji, label, color = "risk-low", "🟢", "LOW RISK", "#22c55e"

# ─────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────
st.markdown("# 📡 Customer Churn Predictor")
st.markdown(
    "*Logistic Regression · Telco dataset · AUC 0.855 · Adjust sidebar to update live*"
)
st.markdown("---")

# KPIs
k1, k2, k3, k4 = st.columns(4)
for col, val, lbl in zip(
    [k1, k2, k3, k4],
    ["7,043", "26.5%", "37.6 mo", "18.0 mo"],
    [
        "Total Customers",
        "Overall Churn Rate",
        "Avg Tenure (Stayed)",
        "Avg Tenure (Churned)",
    ],
):
    col.markdown(
        f'<div class="metric-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")

# ── Left: prediction ──────────────────────────────────────────
with left:
    st.markdown("### Prediction Result")

    st.markdown(
        f"""<div class="{cls}">
          <div class="risk-title">{emoji} Churn Probability</div>
          <div class="risk-pct" style="color:{color}">{pct:.1f}%</div>
          <div class="risk-sub">{label} · Adjust inputs in the sidebar</div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Gauge bar
    st.markdown("<br>", unsafe_allow_html=True)
    fill = min(int(pct), 100)
    st.markdown(
        f"""<div class="bar-wrap">
          <div class="bar-label"><span>0%</span>
            <span style="color:{color};font-weight:600">{pct:.1f}%</span>
            <span>100%</span></div>
          <div class="bar-bg"><div class="bar-fill" style="width:{fill}%;background:{color}"></div></div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Feature contributions
    st.markdown("<br>**Top Churn Drivers for This Customer**")
    top = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    max_abs = max(abs(v) for _, v in top) or 1
    for feat, val in top:
        bar_color = "#ef4444" if val > 0 else "#22c55e"
        bw = int(abs(val) / max_abs * 100)
        arrow = "▲ increases risk" if val > 0 else "▼ reduces risk"
        st.markdown(
            f"""<div class="bar-wrap">
              <div class="bar-label">
                <span>{feat}</span>
                <span style="color:{bar_color};font-size:0.68rem">{arrow}</span>
              </div>
              <div class="bar-bg">
                <div class="bar-fill" style="width:{bw}%;background:{bar_color}"></div>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

# ── Right: dataset insights ───────────────────────────────────
with right:
    st.markdown("### Dataset Benchmarks")

    def insight_bars(title, data, selected, accent):
        st.markdown(f"**{title}**")
        for k, v in data.items():
            w = int(v * 100)
            hl = (
                f"border-left:3px solid {accent};padding-left:6px;"
                if k == selected
                else ""
            )
            st.markdown(
                f"""<div class="bar-wrap" style="{hl}">
                  <div class="bar-label"><span>{k}</span><span>{v:.0%}</span></div>
                  <div class="bar-bg">
                    <div class="bar-fill" style="width:{w}%;background:{accent}"></div>
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )
        st.markdown("<br>", unsafe_allow_html=True)

    insight_bars(
        "Churn Rate by Contract", STATS["churn_by_contract"], contract, "#38bdf8"
    )
    insight_bars(
        "Churn Rate by Internet", STATS["churn_by_internet"], internet, "#a78bfa"
    )
    insight_bars(
        "Churn Rate by Payment Method", STATS["churn_by_payment"], payment, "#fb923c"
    )

    # Contextual tips
    tips = []
    if contract == "Month-to-month":
        tips.append(
            "📋 Month-to-month customers churn at **4×** the rate of two-year contracts. A loyalty discount could flip retention."
        )
    if internet == "Fiber optic":
        tips.append(
            "🌐 Fiber optic has the highest churn (42%). Pricing sensitivity and service quality are key pain points."
        )
    if payment == "Electronic check":
        tips.append(
            "💳 Electronic check payers churn at 45% — switching to auto-pay is strongly correlated with lower churn."
        )
    if tenure < 12:
        tips.append(
            "📅 New customers (tenure < 12 mo) are at peak churn risk. Early onboarding programs matter most."
        )

    if tips:
        st.markdown("**💡 Retention Insights**")
        for tip in tips:
            st.info(tip)

st.markdown("---")
st.caption(
    "Model: Logistic Regression · Dataset: Telco Customer Churn (IBM) · AUC-ROC 0.855 · Predictions are probabilistic estimates."
)
