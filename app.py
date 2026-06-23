import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Premium Page Configuration
st.set_page_config(
    page_title="MSME Underwriting Portal | Institutional Risk Management", 
    page_icon="🏢", 
    layout="wide"
)

# 2. Advanced CSS Injection: Blurred Backdrop & Premium Midnight/Gold Palette
st.markdown("""
    <style>
        /* 1. Base Wallpaper Layer with Deep Filter & 50% Visual Blur Effect */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(rgba(10, 17, 32, 0.88), rgba(10, 17, 32, 0.88)), 
                        url("https://images.unsplash.com/photo-1639762681485-074b7f938ba0?q=80&w=2000&auto=format&fit=crop");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            backdrop-filter: blur(12px); /* Delivers smooth structural focus shift */
            -webkit-backdrop-filter: blur(12px);
        }
        
        /* Clean out default white background wrappers */
        [data-testid="stHeader"], .main, [data-testid="stSidebar"] {
            background-color: transparent !important;
        }

        /* 2. Custom Typography Overrides */
        h1 {
            color: #f59e0b !important; /* Premium Dark Yellow / Amber Gold */
            font-family: 'Inter', -apple-system, sans-serif;
            font-weight: 800 !important;
            letter-spacing: -0.025em;
        }
        h2, h3, h4 {
            color: #f3f4f6 !important; /* Clean Ice-Grey Headers */
            font-family: 'Inter', sans-serif;
            font-weight: 600 !important;
        }
        span, p, label {
            color: #e5e7eb !important; /* High-contrast Silver-Grey labels */
        }
        
        /* 3. Slate Grey Container Cards for Input Focus */
        .metric-card {
            background-color: rgba(30, 41, 59, 0.75); /* Dark Slate Grey with Opacity */
            border: 1px solid rgba(71, 85, 105, 0.5);
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
            margin-bottom: 20px;
        }
        
        /* 4. Action Button - Premium Amber Gold Overrides */
        div.stButton > button:first-child {
            background-color: #d97706 !important; /* Deep Dark Yellow / Gold */
            color: #0f172a !important; /* Crisp Midnight Blue text */
            border-radius: 6px !important;
            border: 1px solid #f59e0b !important;
            padding: 12px 24px !important;
            font-weight: 700 !important;
            width: 100% !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            transition: all 0.2s ease;
        }
        div.stButton > button:first-child:hover {
            background-color: #f59e0b !important;
            box-shadow: 0 0 15px rgba(245, 158, 11, 0.4) !important;
            transform: translateY(-1px);
        }
        
        /* 5. Custom Verdict Callout Banners */
        .verdict-approved {
            background-color: rgba(16, 185, 129, 0.15);
            border: 1px solid #10b981;
            border-left: 6px solid #10b981;
            padding: 20px;
            border-radius: 8px;
            color: #34d399 !important;
        }
        .verdict-rejected {
            background-color: rgba(239, 68, 68, 0.15);
            border: 1px solid #ef4444;
            border-left: 6px solid #ef4444;
            padding: 20px;
            border-radius: 8px;
            color: #f87171 !important;
        }
    </style>
""", unsafe_allow_html=True)

# 3. Secure Asset Initialization
@st.cache_resource
def load_assets():
    model = joblib.load('credit_risk_logistic_model.joblib')
    features = joblib.load('model_features.joblib')
    return model, features

try:
    model, model_features = load_assets()
    st.sidebar.markdown("### 🟢 Node Status")
    st.sidebar.caption("Connected to Risk Core Gateway V2.4")
except Exception as e:
    st.sidebar.error("System Core Offline")
    st.stop()

# 4. Main Executive Header Layout
st.markdown("<span style='color: #d97706; font-weight: 700; letter-spacing: 0.05em;'>🏢 RISK MANAGEMENT SOLUTIONS | INSTITUTIONAL DIVISION</span>", unsafe_allow_html=True)
st.title("MSME Credit Risk Underwriting Terminal")
st.markdown("<p style='color: #9ca3af !important;'>Commercial Risk Matrix Evaluation Engine • Production Node Server</p>", unsafe_allow_html=True)
st.markdown("<hr style='margin-top: 10px; margin-bottom: 30px; border-color: rgba(71, 85, 105, 0.3);'>", unsafe_allow_html=True)

# 5. Split Dashboard Layout into 2-Column Workspaces
left_col, right_col = st.columns([1.1, 0.9], gap="large")

with left_col:
    st.markdown("### 📋 Application Parameters")
    st.caption("Complete all regulatory financial metrics to calculate risk probability profiles.")
    
    # Input data card wrapper
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        loan_amount_inr = st.number_input("Requested Capital Exposure (INR)", min_value=1000, value=250000, step=10000)
        duration_months = st.number_input("Amortization Period (Months)", min_value=1, max_value=72, value=24)
        age_years = st.number_input("Principal Applicant Age (Years)", min_value=18, max_value=100, value=35)
    with sub_col2:
        installment_income_pct = st.slider("Debt Service Installment (% of Income)", min_value=1, max_value=10, value=4)
        sector_selection = st.selectbox("Macro Economic Sector Target", [
            "Infrastructure / Heavy Industry",
            "Textiles / Manufacturing",
            "Retail Trade & Commerce",
            "Services & IT Professional",
            "Agriculture & Allied Activities"
        ])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process inputs mapping structures
    sector_mapping = {
        "Infrastructure / Heavy Industry": 12.4,
        "Textiles / Manufacturing": 9.2,
        "Retail Trade & Commerce": 5.8,
        "Services & IT Professional": 3.1,
        "Agriculture & Allied Activities": 8.4
    }
    rbi_sector_gnpa_pct = sector_mapping[sector_selection]
    credit_amount_dm = loan_amount_inr / 28
    debt_burden_score = installment_income_pct * duration_months
    age_to_tenure_ratio = age_years / duration_months
    
    input_data = {
        "duration_months": duration_months, "credit_amount_DM": credit_amount_dm,
        "installment_income_pct": installment_income_pct, "age_years": age_years,
        "loan_amount_INR": loan_amount_inr, "rbi_sector_gnpa_pct": rbi_sector_gnpa_pct,
        "debt_burden_score": debt_burden_score, "age_to_tenure_ratio": age_to_tenure_ratio
    }
    input_df = pd.DataFrame([input_data])[model_features]
    
    # Big Action Button
    calculate_clicked = st.button("🚀 Run Risk Underwriting Audit")

with right_col:
    st.markdown("### 📊 Live Evaluation Vector")
    st.caption("Risk verdicts display dynamically here upon system audit trigger execution.")
    
    if calculate_clicked:
        risk_probability = model.predict_proba(input_df)[0, 1]
        
        # Display elegant internal card layout for analytical outputs
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("#### Portfolio Risk Scoring Matrix Metrics")
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric(label="Computed Default Risk", value=f"{risk_probability * 100:.2f}%")
        with m_col2:
            st.metric(label="Risk Tolerance Cap", value="35.00%")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 6. Apply Premium Custom CSS Banner Alerts
        if risk_probability >= 0.35:
            st.markdown(f"""
                <div class="verdict-rejected">
                    <h3 style="color: #f87171 !important; margin-top: 0;">❌ RISK LEVEL EXCEEDED: APPLICATION DECLINED</h3>
                    <p style="margin-top: 8px; margin-bottom: 0; color: #fca5a5 !important;"><b>Underwriting Operational Assessment:</b> 
                    The calculated risk probability profile exceeds institutional risk appetite boundaries. 
                    Macro sector GNPA stress indicators (<b>{rbi_sector_gnpa_pct}%</b>) combined with borrower debt ratios present an outside variance vector.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="verdict-approved">
                    <h3 style="color: #34d399 !important; margin-top: 0;">✅ RISK RATIO APPROVED: CREDIT VERDICT PASS</h3>
                    <p style="margin-top: 8px; margin-bottom: 0; color: #a7f3d0 !important;"><b>Underwriting Operational Assessment:</b> 
                    Applicant metrics reside comfortably inside internal capital asset preservation baselines. 
                    Cash flow metrics validate creditworthiness guidelines.</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        # State display placeholder for an untouched clean load
        st.markdown("""
            <div style="border: 2px dashed rgba(148, 163, 184, 0.3); padding: 50px; text-align: center; border-radius: 12px; color: #94a3b8; margin-top: 20px; background-color: rgba(30, 41, 59, 0.4);">
                ⚙️ Engine Idle. Fill in application parameters and execute audit routine to generate corporate ledger decision.
            </div>
        """, unsafe_allow_html=True)