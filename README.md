# 📡 Customer Churn Prediction Dashboard
 
A Streamlit web application that predicts customer churn in real-time using a Logistic Regression model trained on the Telco Customer Churn dataset. No file upload required — the model is fully embedded in the app.
 
---
 
## 🖥️ Live Demo
 
> Adjust any customer attribute in the sidebar and the churn probability updates instantly.
 
![Dashboard Preview](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
 
---
 
## 📁 Project Structure
 
```
customer-churn-prediction/
│
├── app.py                  # Streamlit prediction dashboard (main entry point)
├── code.ipynb              # Full EDA + model training notebook
├── CustomerChurn.csv       # Source dataset (Telco, IBM)
└── README.md
```
 
---
 
## 🚀 Getting Started
 
### 1. Clone the repository
 
```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```
 
### 2. Install dependencies
 
```bash
pip install streamlit
```
 
> `app.py` has **no other runtime dependencies** — the trained model is embedded as coefficients directly in the file.
 
### 3. Run the app
 
```bash
streamlit run app.py
```
 
Open your browser at `http://localhost:8501`.
 
---
 
## ☁️ Deploy on Streamlit Cloud
 
1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, set **Main file path** to `app.py`.
4. Click **Deploy** — no `requirements.txt` needed beyond `streamlit`.
---
 
## 🧠 Model Details
 
| Property | Value |
|---|---|
| Algorithm | Logistic Regression |
| Training data | 7,043 customers (Telco IBM dataset) |
| Class balancing | Oversampling (minority upsampling) |
| AUC-ROC | **0.855** |
| Features | 19 raw + 3 engineered |
| Runtime dependency | None (coefficients embedded) |
 
### Engineered Features
 
| Feature | Description |
|---|---|
| `charge_per_tenure` | Monthly charges ÷ (tenure + 1) |
| `has_streaming` | 1 if Streaming TV or Movies is active |
| `has_security` | 1 if Online Security or Device Protection is active |
 
---
 
## 📊 Dashboard Features
 
- **Live churn probability** — updates on every sidebar change, no button click required
- **Risk badge** — colour-coded High / Medium / Low verdict with exact percentage
- **Churn driver bars** — top 8 features pushing risk up (🔴) or down (🟢) for the current customer
- **Dataset benchmarks** — churn rates by Contract, Internet Service, and Payment Method, with the customer's current selection highlighted
- **Contextual retention tips** — actionable insights that appear only when relevant (e.g. month-to-month contract, electronic check payment, fiber optic, new customer)
---
 
## 📈 Key Insights from the Dataset
 
| Segment | Churn Rate |
|---|---|
| Month-to-month contract | 42.7% |
| Fiber optic internet | 41.9% |
| Electronic check payment | 45.3% |
| Two-year contract | 2.8% |
| Tenure > 36 months | ~15% |
 
---
 
## 🔬 Notebook (`code.ipynb`)
 
The notebook covers the full ML pipeline:
 
1. **Data loading & exploration** — shape, dtypes, missing values, class distribution
2. **EDA** — churn distribution, box plots, churn-by-category bar charts
3. **Preprocessing** — label encoding, TotalCharges imputation, feature engineering
4. **Class balancing** — minority class upsampling
5. **Model training** — Logistic Regression, Decision Tree, Random Forest, XGBoost
6. **Evaluation** — classification report, ROC curves, confusion matrices
7. **Feature importance** — Random Forest importances, XGBoost gain scores
---
 
## 📦 Dataset
 
**Telco Customer Churn** — IBM Sample Dataset  
7,043 customers · 21 features · Binary target (`Churn`: Yes / No)
 
Key columns: `tenure`, `Contract`, `InternetService`, `MonthlyCharges`, `TotalCharges`, `PaymentMethod`, and 15 service/demographic features.
 
---
 
## 🛠️ Tech Stack
 
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189FDA?style=flat)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
 
---
