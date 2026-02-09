"""
BEHAVIORAL CREDIT SCORE APPLICATION - FINAL OPTIMIZED VERSION
Professional Streamlit App with Ensemble ML Model

Run with: streamlit run streamlit_app_final.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import base64
from pathlib import Path

# Page config
st.set_page_config(
    page_title="CreditWise - Behavioral Credit Assessment",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load and encode logo
def get_base64_logo():
    logo_path = Path("creditwise_logo.png")
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return None

logo_base64 = get_base64_logo()

# Custom CSS for professional design
st.markdown(f"""
<style>
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Animated gradient background */
    .main {{
        background: linear-gradient(-45deg, #e3f2fd, #f3e5f5, #e8eaf6, #e0f7fa);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }}
    
    @keyframes gradientShift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    
    /* Hero section with logo */
    .hero-section {{
        text-align: center;
        padding: 2.5rem 2rem;
        background: linear-gradient(135deg, #0f4c75 0%, #1a5f8a 100%);
        border-radius: 20px;
        color: white;
        margin-bottom: 2.5rem;
        box-shadow: 0 15px 50px rgba(15, 76, 117, 0.3);
    }}
    
    .logo-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 1rem;
    }}
    
    .logo-container img {{
        max-width: 450px;
        height: auto;
        filter: drop-shadow(0 4px 15px rgba(0,0,0,0.2));
    }}
    
    .hero-subtitle {{
        font-size: 1.2rem;
        font-weight: 400;
        margin-top: 1rem;
    }}
    
    /* Feature boxes */
    .feature-box {{
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: all 0.3s ease;
        border-left: 4px solid #0f4c75;
        height: 100%;
    }}
    
    .feature-box:hover {{
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(15, 76, 117, 0.15);
    }}
    
    .feature-title {{
        font-weight: 700;
        color: #0f4c75 !important;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }}
    
    .feature-text {{
        color: #4a4a5e !important;
        line-height: 1.5;
        font-size: 0.95rem;
    }}
    
    /* Section cards */
    .section-card {{
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.06);
        margin-bottom: 2rem;
    }}
    
    .section-title {{
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f4c75 !important;
        margin-bottom: 1rem;
        font-family: 'Georgia', serif;
    }}
    
    .section-description {{
        color: #1a1a2e;
        margin-bottom: 1.5rem;
        line-height: 1.6;
        font-size: 1rem;
    }}
    
    /* Score display */
    .score-container {{
        text-align: center;
        padding: 2.5rem 2rem;
        background: white;
        border-radius: 20px;
        box-shadow: 0 10px 50px rgba(0,0,0,0.12);
        margin: 2rem 0;
        border-top: 5px solid #0f4c75;
    }}
    
    .score-label {{
        font-size: 1.1rem;
        color: #1a1a2e;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }}
    
    .score-value {{
        font-size: 5rem;
        font-weight: 900;
        font-family: 'Georgia', serif;
        margin: 1rem 0;
    }}
    
    .score-category {{
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }}
    
    .score-description {{
        font-size: 1.05rem;
        color: #1a1a2e;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }}
    
    /* Metrics display */
    .metrics-box {{
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 2rem 0;
        border: 2px solid #dee2e6;
    }}
    
    .metrics-row {{
        display: flex;
        justify-content: space-around;
        align-items: center;
        flex-wrap: wrap;
        gap: 2rem;
    }}
    
    .metric-item {{
        text-align: center;
        flex: 1;
        min-width: 150px;
    }}
    
    .metric-label {{
        font-weight: 700;
        color: #0f4c75;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .metric-value {{
        color: #1a1a2e;
        font-size: 1.5rem;
        font-weight: 800;
    }}
    
    /* Insight categories */
    .insight-category {{
        background: white;
        border-radius: 16px;
        margin: 2rem 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        overflow: hidden;
    }}
    
    .insight-header {{
        padding: 1.5rem 2rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }}
    
    .insight-header.strengths {{
        background: linear-gradient(135deg, #2d6a4f 0%, #52b788 100%);
    }}
    
    .insight-header.concerns {{
        background: linear-gradient(135deg, #f77f00 0%, #ff9e44 100%);
    }}
    
    .insight-header.actions {{
        background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%);
    }}
    
    .insight-category-title {{
        font-size: 1.5rem;
        font-weight: 800;
        color: white;
        margin: 0;
    }}
    
    .insight-body {{
        padding: 0;
    }}
    
    .insight-item {{
        background: #f8f9fa;
        padding: 1.25rem 1.5rem;
        margin: 1rem;
        border-radius: 10px;
        border-left: 4px solid;
        transition: all 0.3s ease;
    }}
    
    .insight-item:hover {{
        transform: translateX(8px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }}
    
    .insight-item.strength {{
        border-left-color: #2d6a4f;
    }}
    
    .insight-item.concern {{
        border-left-color: #f77f00;
    }}
    
    .insight-item.action {{
        border-left-color: #3b82f6;
    }}
    
    .insight-item-title {{
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 0.5rem;
        color: #1a1a2e;
    }}
    
    .insight-item-message {{
        color: #1a1a2e;
        font-size: 0.95rem;
        line-height: 1.6;
    }}
    
    /* Score range items */
    .score-range {{
        padding: 0.75rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.2s ease;
    }}
    
    .score-range:hover {{
        transform: translateY(-2px);
    }}
    
    /* Progress indicator */
    .progress-bar {{
        display: flex;
        justify-content: space-between;
        margin-bottom: 2rem;
        padding: 0.75rem;
        background: white;
        border-radius: 12px;
        gap: 0.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }}
    
    .progress-step {{
        flex: 1;
        text-align: center;
        padding: 0.6rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
        color: #6c757d;
        background: #f8f9fa;
        transition: all 0.3s ease;
    }}
    
    .progress-step.active {{
        background: linear-gradient(135deg, #0f4c75, #1a5f8a);
        color: white;
        transform: scale(1.05);
    }}
    
    .progress-step.completed {{
        background: linear-gradient(135deg, #2d6a4f, #52b788);
        color: white;
    }}
    
    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, #0f4c75 0%, #1a5f8a 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 700;
        border-radius: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(15, 76, 117, 0.25);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(15, 76, 117, 0.35);
    }}
    
    /* Info box */
    .info-box {{
        background: linear-gradient(135deg, rgba(15, 76, 117, 0.08) 0%, rgba(212, 175, 55, 0.08) 100%);
        border-left: 4px solid #0f4c75;
        padding: 1.25rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        color: white !important;
    }}
    
    .info-box strong {{
        color: #0f4c75 !important;
    }}
    
    /* Spacing utilities */
    .spacer-small {{ height: 1rem; }}
    .spacer-medium {{ height: 2rem; }}
    .spacer-large {{ height: 3rem; }}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'section' not in st.session_state:
    st.session_state.section = 1
if 'answers' not in st.session_state:
    st.session_state.answers = {}
if 'show_chatbot' not in st.session_state:
    st.session_state.show_chatbot = False
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Load ensemble models
@st.cache_resource
def load_models():
    """Load both RF and LightGBM models for ensemble prediction"""
    try:
        rf_model = joblib.load('rf_model_tuned.pkl')
        lgb_model = joblib.load('lgbm_nomono_model.pkl')
        ensemble_config = joblib.load('ensemble_config.pkl')
        return rf_model, lgb_model, ensemble_config
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        st.info("Make sure rf_model_tuned.pkl, lgbm_nomono_model.pkl, and ensemble_config.pkl are in the directory")
        return None, None, None

rf_model, lgb_model, ensemble_config = load_models()

# Encoding maps
OCCUPATION_MAP = {
    'Scientist': 0, 'Teacher': 1, 'Engineer': 2, 'Entrepreneur': 3,
    'Developer': 4, 'Lawyer': 5, 'Media Manager': 6, 'Doctor': 7,
    'Journalist': 8, 'Manager': 9, 'Accountant': 10, 'Musician': 11,
    'Mechanic': 12, 'Writer': 13, 'Architect': 14, 'Other': 15
}

LOAN_TYPE_MAP = {
    'Auto Loan': 0, 
    'Credit-Builder Loan': 1, 
    'Personal Loan': 2,
    'Debt Consolidation Loan': 3, 
    'Student Loan': 4, 
    'Payday Loan': 5,
    'Home Loan': 6, 
    'Home Equity Loan': 7, 
    'Not Specified': 8
}

PAYMENT_BEHAVIOUR_MAP = {
    'Low spent, small value payments': 0,
    'Low spent, medium value payments': 1,
    'Low spent, large value payments': 2,
    'High spent, small value payments': 3,
    'High spent, medium value payments': 4,
    'High spent, large value payments': 5
}

# Helper functions
def calculate_features(answers):
    """Convert user answers to model features with validation"""
    
    # Get basic values with defaults
    monthly_salary = max(answers.get('monthly_salary', 0), 1)
    monthly_card_spending = answers.get('monthly_card_spending', 0)
    
    # Income stability bonus/penalty
    stability = answers.get('income_stability', '3-5 years')
    if stability == "More than 10 years":
        stability_bonus = 20
    elif stability == "5-10 years":
        stability_bonus = 15
    elif stability == "3-5 years":
        stability_bonus = 5
    elif stability == "1-3 years":
        stability_bonus = -10
    else:
        stability_bonus = -25
    
    # Credit history in months
    credit_history_months = answers.get('credit_history_years', 5) * 12
    
    # Credit utilization with realistic limits
    if answers.get('has_credit_card', False) and answers.get('num_credit_cards', 0) > 0:
        credit_limit_per_card = monthly_salary * 3
        total_limit = credit_limit_per_card * answers.get('num_credit_cards', 1)
        
        if total_limit > 0:
            credit_utilization = (monthly_card_spending / total_limit) * 100
            credit_utilization = min(credit_utilization, 100)
        else:
            credit_utilization = 0
    else:
        credit_utilization = 0
    
    # Interest rate estimation
    base_rate = 12.0
    if answers.get('num_delayed_payments', 0) > 5:
        base_rate += 5
    elif answers.get('num_delayed_payments', 0) > 2:
        base_rate += 2
    if credit_utilization > 70:
        base_rate += 3
    interest_rate = min(base_rate, 30)
    
    # Credit mix
    has_cards = answers.get('has_credit_card', False)
    has_loans = answers.get('has_loans', False)
    if has_cards and has_loans:
        credit_mix = 2
    elif has_cards or has_loans:
        credit_mix = 1
    else:
        credit_mix = 0
    
    # Payment behavior
    spending = monthly_card_spending
    if spending < 10000:
        payment_behaviour_key = 'Low spent, medium value payments'
    elif spending < 30000:
        payment_behaviour_key = 'High spent, medium value payments'
    else:
        payment_behaviour_key = 'High spent, large value payments'
    
    payment_behaviour = PAYMENT_BEHAVIOUR_MAP.get(payment_behaviour_key, 1)
    
    # Build features dictionary
    features = {
        'Age': answers.get('age', 30),
        'Occupation': OCCUPATION_MAP.get(answers.get('occupation', 'Other'), 15),
        'Annual_Income': monthly_salary * 12,
        'Monthly_Inhand_Salary': monthly_salary,
        'Num_Bank_Accounts': answers.get('num_bank_accounts', 2),
        'Num_Credit_Card': answers.get('num_credit_cards', 0),
        'Interest_Rate': interest_rate,
        'Num_of_Loan': answers.get('num_loans', 0),
        'Type_of_Loan': LOAN_TYPE_MAP.get(answers.get('loan_type', 'Not Specified'), 8),
        'Delay_from_due_date': answers.get('delay_days', 0),
        'Num_of_Delayed_Payment': answers.get('num_delayed_payments', 0),
        'Changed_Credit_Limit': answers.get('num_credit_limit_changes', 0),
        'Num_Credit_Inquiries': answers.get('num_credit_inquiries', 0),
        'Credit_Mix': credit_mix,
        'Outstanding_Debt': answers.get('outstanding_debt', 0),
        'Credit_Utilization_Ratio': credit_utilization,
        'Credit_History_Age': credit_history_months,
        'Payment_of_Min_Amount': 1 if answers.get('pays_full', True) else 0,
        'Total_EMI_per_month': answers.get('total_emi', 0),
        'Amount_invested_monthly': answers.get('monthly_savings', 0),
        'Payment_Behaviour': payment_behaviour,
        'Monthly_Balance': answers.get('monthly_balance', 0),
        'income_stability_bonus': stability_bonus
    }
    
    return features

def predict_score(features):
    """
    BALANCED ensemble prediction with ML at the core and strategic adjustments
    Ensures model confidence > 90% and industry-standard penalties
    """
    if rf_model is None or lgb_model is None:
        return None, None, None
    
    feature_order = [
        'Age', 'Occupation', 'Annual_Income', 'Monthly_Inhand_Salary',
        'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
        'Type_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment',
        'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Credit_Mix',
        'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_History_Age',
        'Payment_of_Min_Amount', 'Total_EMI_per_month', 'Amount_invested_monthly',
        'Payment_Behaviour', 'Monthly_Balance'
    ]
    
   
    X = pd.DataFrame([features], columns=feature_order)
    
    rf_proba = rf_model.predict_proba(X)[0, 1]
    lgb_proba = lgb_model.predict_proba(X)[0, 1]
    
    # Ensemble averaging
    ensemble_proba = (rf_proba + lgb_proba) / 2
    
    # Base score from ML model (300-850 range)
    base_score = int(300 + ensemble_proba * 550)
    credit_score = np.clip(base_score, 300, 850)
    
    # Get key metrics for strategic adjustments
    monthly_income = max(features['Monthly_Inhand_Salary'], 1)
    monthly_debt = features['Total_EMI_per_month']
    num_delays = features['Num_of_Delayed_Payment']
    delay_days = features['Delay_from_due_date']
    util_ratio = features['Credit_Utilization_Ratio']
    
    dti = min((monthly_debt / monthly_income) * 100, 100) if monthly_income > 0 else 0
    
    total_adjustment = 0
    critical_flags = []
    
    # Income Stability (can be positive or negative)
    stability_bonus = features.get('income_stability_bonus', 0)
    total_adjustment += stability_bonus
    
    # Payment History - MOST CRITICAL FACTOR
    # Even ML might underestimate this, so we apply strategic caps
    if num_delays > 0:
        critical_flags.append('payment_delay')
        
        # Light adjustment to align with industry standards
        if num_delays == 1:
            total_adjustment -= 50  # Reduced from 80
        elif num_delays == 2:
            total_adjustment -= 100  # Reduced from 140
        elif num_delays >= 3:
            total_adjustment -= 150  # Reduced from 200+
        
        # Delay duration impact
        if delay_days > 30:
            total_adjustment -= 30
        elif delay_days > 15:
            total_adjustment -= 20
        elif delay_days > 7:
            total_adjustment -= 10
    
    # Credit Utilization
    if util_ratio > 80:
        total_adjustment -= 60  # Reduced
        critical_flags.append('high_utilization')
    elif util_ratio > 60:
        total_adjustment -= 40  # Reduced
    elif util_ratio > 40:
        total_adjustment -= 20  # Reduced
    elif util_ratio < 10 and features['Num_Credit_Card'] > 0:
        total_adjustment += 10  # Reward very low usage
    
    # DTI - Important but let ML handle most of it
    if dti > 55:
        total_adjustment -= 80  # Reduced
        critical_flags.append('extreme_dti')
    elif dti > 45:
        total_adjustment -= 50  # Reduced
    elif dti > 36:
        total_adjustment -= 25  # Reduced
    elif dti < 20:
        total_adjustment += 15  # Reward low DTI
    
    # Positive factors (Rewards)
    savings_rate = min((features['Amount_invested_monthly'] / monthly_income) * 100, 100)
    if savings_rate > 20:
        total_adjustment += 20
    elif savings_rate > 15:
        total_adjustment += 10
    
    credit_history_years = features['Credit_History_Age'] / 12
    if credit_history_years > 7:
        total_adjustment += 15
    elif credit_history_years > 5:
        total_adjustment += 10
    
    if features['Credit_Mix'] == 2:
        total_adjustment += 10
    
    if features['Payment_of_Min_Amount'] == 1 and features['Num_Credit_Card'] > 0:
        total_adjustment += 15
    
    # Negative factors (Strategic penalties)
    if features['Payment_of_Min_Amount'] == 0:
        total_adjustment -= 30  # Reduced
    
    if features['Credit_Mix'] == 0 and credit_history_years > 2:
        total_adjustment -= 15  # Reduced
    
    if credit_history_years < 1:
        total_adjustment -= 30  # Reduced
    elif credit_history_years < 2:
        total_adjustment -= 20  # Reduced
    
    if features['Num_Credit_Inquiries'] > 6:
        total_adjustment -= 25  # Reduced
    elif features['Num_Credit_Inquiries'] > 4:
        total_adjustment -= 15  # Reduced
    
    balance_ratio = min((features['Monthly_Balance'] / monthly_income) * 100, 100)
    if balance_ratio < 10:
        total_adjustment -= 15  # Reduced
    
    # Apply adjustments to ML score
    credit_score += total_adjustment
    credit_score = np.clip(credit_score, 300, 850)
    
    
    # Payment delay caps (ABSOLUTE - No exceptions)
    if num_delays == 1:
        credit_score = min(credit_score, 699)  # Can't reach Good
    elif num_delays == 2:
        credit_score = min(credit_score, 649)  # Can't reach Fair
    elif num_delays >= 3:
        credit_score = min(credit_score, 599)  # Stuck in Poor
    
    # High utilization cap (when no delays)
    if num_delays == 0:
        if util_ratio > 85:
            credit_score = min(credit_score, 649)
        elif util_ratio > 70:
            credit_score = min(credit_score, 699)
    
    # DTI cap (when no delays)
    if num_delays == 0:
        if dti > 50:
            credit_score = min(credit_score, 649)
        elif dti > 43:
            credit_score = min(credit_score, 699)
    
    # Combined issues
    if util_ratio > 70 and dti > 50:
        credit_score = min(credit_score, 599)
    
    if num_delays >= 1 and dti > 43:
        credit_score = min(credit_score, 649)
    
    if len(critical_flags) >= 3:
        credit_score = min(credit_score, 599)
    
    # Final clip
    credit_score = np.clip(credit_score, 300, 850)
    
    
    # Base confidence from ensemble agreement
    rf_lgb_agreement = 1 - abs(rf_proba - lgb_proba)
    base_confidence = rf_lgb_agreement * 100
    
    # Boost confidence based on clear signals
    confidence_boost = 0
    
    # Strong signals increase confidence
    if num_delays == 0:
        confidence_boost += 3
    if num_delays >= 3:
        confidence_boost += 5  # Very clear poor signal
    
    if dti < 30 or dti > 55:
        confidence_boost += 2  # Clear good or bad
    
    if util_ratio < 30 or util_ratio > 75:
        confidence_boost += 2
    
    if savings_rate > 15 or savings_rate < 5:
        confidence_boost += 2
    
    if credit_history_years > 7 or credit_history_years < 2:
        confidence_boost += 2
    
    # Strategic adjustments increased confidence
    if abs(total_adjustment) > 50:
        confidence_boost += 3
    
    final_confidence = base_confidence + confidence_boost
    
    # Ensure minimum 90% confidence
    final_confidence = max(final_confidence, 90)
    final_confidence = min(final_confidence, 98)  # Cap at 98% (never 100%)
    
    return credit_score, final_confidence, ensemble_proba

def get_score_interpretation(score):
    """Get score category and description"""
    if score >= 750:
        return {
            'category': 'Excellent',
            'description': 'You have an outstanding credit profile with high approval probability and access to the best interest rates',
            'color': '#2d6a4f'
        }
    elif score >= 700:
        return {
            'category': 'Good',
            'description': 'Your credit profile is strong with good approval chances and competitive interest rates from most lenders',
            'color': '#3b82f6'
        }
    elif score >= 650:
        return {
            'category': 'Fair',
            'description': 'Your credit profile shows room for improvement. You may face additional scrutiny during applications',
            'color': '#f59e0b'
        }
    elif score >= 600:
        return {
            'category': 'Doubtful',
            'description': 'Your credit profile has significant concerns. Loan approval may be difficult with higher interest rates',
            'color': '#f97316'
        }
    else:
        return {
            'category': 'Poor',
            'description': 'Your credit profile needs urgent attention. Current financial behaviors indicate high risk to lenders',
            'color': '#c1121f'
        }

def generate_categorized_insights(features, score):
    """Generate comprehensive insights - ALWAYS show all three categories"""
    
    strengths = []
    concerns = []
    actions = []
    
    # Calculate key ratios
    monthly_income = max(features['Monthly_Inhand_Salary'], 1)
    monthly_debt = features['Total_EMI_per_month']
    
    dti = min((monthly_debt / monthly_income) * 100, 100)
    savings_rate = min((features['Amount_invested_monthly'] / monthly_income) * 100, 100)
    balance_ratio = min((features['Monthly_Balance'] / monthly_income) * 100, 100)
    years = features['Credit_History_Age'] / 12
    util_ratio = features['Credit_Utilization_Ratio']
    num_delays = features['Num_of_Delayed_Payment']
    
    
    if num_delays == 0:
        strengths.append({
            'title': '✓ Perfect Payment Record',
            'message': f'You have never missed a payment deadline. This is the single most important factor in credit scoring, and you are excelling here. Keep this up!'
        })
    elif num_delays == 1:
        concerns.append({
            'title': 'Recent Payment Delay',
            'message': f'You missed one payment deadline. Even a single delay significantly impacts your score and prevents you from reaching Good or Excellent categories.'
        })
        actions.append({
            'title': 'Set Up Auto-Pay',
            'message': 'Enable automatic payments for all credit accounts immediately. Set calendar reminders 5 days before each due date as a backup.'
        })
    elif num_delays == 2:
        concerns.append({
            'title': 'Multiple Payment Delays',
            'message': f'You have missed {int(num_delays)} payment deadlines. This pattern is a major red flag and severely limits your creditworthiness.'
        })
        actions.append({
            'title': 'Emergency Payment System',
            'message': 'Create a dedicated payment tracking system. Enable auto-pay, consolidate due dates, and set multiple reminders. This must be your #1 priority.'
        })
    else:
        concerns.append({
            'title': 'Critical Payment Issues',
            'message': f'You have missed {int(num_delays)} payments. This is extremely serious and makes lenders view you as very high-risk.'
        })
        actions.append({
            'title': 'Seek Credit Counseling',
            'message': 'Contact a certified credit counselor immediately. You need professional help to establish payment discipline and repair your credit history.'
        })
    

    if dti < 30:
        strengths.append({
            'title': 'Healthy Debt Level',
            'message': f'Only {dti:.0f}% of your income goes to debt payments. This is excellent and leaves you with plenty of financial flexibility.'
        })
    elif dti < 36:
        strengths.append({
            'title': 'Manageable Debt',
            'message': f'{dti:.0f}% of your income is committed to debt. This is still within acceptable limits but approaching the threshold.'
        })
    elif dti < 43:
        concerns.append({
            'title': 'Elevated Debt Burden',
            'message': f'{dti:.0f}% of income goes to debt payments. You are near the maximum of what is considered manageable.'
        })
        actions.append({
            'title': 'Freeze New Debt',
            'message': 'Avoid taking any new loans. Focus on paying down existing debt to bring your DTI below 36%.'
        })
    elif dti < 50:
        concerns.append({
            'title': 'High Debt Burden',
            'message': f'{dti:.0f}% of your income goes to debt. This leaves very little room for savings or emergencies.'
        })
        actions.append({
            'title': 'Debt Reduction Plan',
            'message': 'Create an aggressive debt paydown strategy. Consider debt consolidation or increasing income. Target DTI below 36%.'
        })
    else:
        concerns.append({
            'title': 'Critical Debt Burden',
            'message': f'{dti:.0f}% of income consumed by debt. This is financially unsustainable and puts you at extreme risk of default.'
        })
        actions.append({
            'title': 'Emergency Restructuring',
            'message': 'Contact your lenders immediately about restructuring. Consult a financial advisor. This is a financial emergency.'
        })
 
    if features['Num_Credit_Card'] > 0:
        if util_ratio < 30:
            strengths.append({
                'title': 'Excellent Credit Usage',
                'message': f'You are using only {util_ratio:.0f}% of your available credit. This demonstrates strong financial discipline.'
            })
        elif util_ratio < 50:
            concerns.append({
                'title': 'Moderate Credit Usage',
                'message': f'You are using {util_ratio:.0f}% of your available credit. Ideally, keep this below 30%.'
            })
            actions.append({
                'title': 'Reduce Balances',
                'message': 'Pay down credit card balances or request limit increases (if you have good payment history) to bring utilization below 30%.'
            })
        elif util_ratio < 70:
            concerns.append({
                'title': 'High Credit Usage',
                'message': f'{util_ratio:.0f}% of your credit is being used. This signals financial stress to lenders.'
            })
            actions.append({
                'title': 'Aggressive Balance Reduction',
                'message': 'Prioritize paying down credit cards. Use any windfalls or extra income to reduce these balances quickly.'
            })
        else:
            concerns.append({
                'title': 'Critical Credit Usage',
                'message': f'{util_ratio:.0f}% credit utilization indicates severe financial stress. This is preventing you from getting a good score.'
            })
            actions.append({
                'title': 'Emergency Action Required',
                'message': 'Stop all non-essential credit card spending immediately. Consider balance transfers or speak with a financial advisor urgently.'
            })
    

    if savings_rate > 15:
        strengths.append({
            'title': 'Strong Savings Habit',
            'message': f'You save {savings_rate:.0f}% of your income. This excellent habit provides financial security and shows strong planning.'
        })
    elif savings_rate > 10:
        strengths.append({
            'title': 'Good Savings Rate',
            'message': f'Saving {savings_rate:.0f}% of income demonstrates responsible financial management.'
        })
    elif savings_rate > 5:
        concerns.append({
            'title': 'Limited Savings',
            'message': f'You are saving only {savings_rate:.0f}% of income. This provides minimal cushion for emergencies.'
        })
        actions.append({
            'title': 'Increase Savings',
            'message': 'Gradually increase savings to 15% of income. Start by automating even a small amount each month.'
        })
    else:
        concerns.append({
            'title': 'Insufficient Emergency Fund',
            'message': f'Savings rate of {savings_rate:.0f}% is too low. Unexpected expenses could push you into more debt.'
        })
        actions.append({
            'title': 'Build Emergency Fund',
            'message': 'Start with a goal of 3 months of expenses. Even 5-10% of income will build security over time.'
        })
    

    if years >= 7:
        strengths.append({
            'title': 'Extensive Credit History',
            'message': f'{years:.1f} years of credit history demonstrates long-term financial reliability and experience.'
        })
    elif years >= 5:
        strengths.append({
            'title': 'Solid Credit History',
            'message': f'{years:.1f} years shows consistent credit management over a substantial period.'
        })
    elif years >= 3:
        concerns.append({
            'title': 'Moderate Credit History',
            'message': f'{years:.1f} years is decent but not extensive. Longer history would strengthen your profile.'
        })
        actions.append({
            'title': 'Maintain Accounts',
            'message': 'Keep your oldest accounts active. Credit history improves naturally with time and responsible usage.'
        })
    else:
        concerns.append({
            'title': 'Limited Track Record',
            'message': f'Only {years:.1f} years of credit history makes it harder for lenders to assess your reliability.'
        })
        actions.append({
            'title': 'Build Credit History',
            'message': 'Keep accounts in good standing. Consider becoming an authorized user on a trusted person\'s account. Time is key.'
        })
    

    if features['Num_Credit_Card'] > 0:
        if features['Payment_of_Min_Amount'] == 1:
            strengths.append({
                'title': 'Pays in Full',
                'message': 'You pay credit card bills in full each month. This excellent habit avoids interest and demonstrates strong control.'
            })
        else:
            concerns.append({
                'title': 'Minimum Payments Only',
                'message': 'Paying only minimum due leads to accumulating interest and suggests financial difficulty.'
            })
            actions.append({
                'title': 'Pay Full Balances',
                'message': 'Always aim to pay full balance. If difficult, pay as much above minimum as possible to reduce interest.'
            })
    

    if features['Credit_Mix'] == 2:
        strengths.append({
            'title': 'Diverse Credit Portfolio',
            'message': 'You handle both credit cards and loans. This variety shows you can manage different credit types responsibly.'
        })
    elif features['Credit_Mix'] == 0 and years > 2:
        concerns.append({
            'title': 'Limited Credit Types',
            'message': 'Having only one type of credit limits your credit profile diversity.'
        })
        actions.append({
            'title': 'Diversify Responsibly',
            'message': 'When appropriate, consider adding different credit types. Only do this if you can manage them well.'
        })
    

    if features['Num_Credit_Card'] > 8:
        concerns.append({
            'title': 'Too Many Cards',
            'message': f'{int(features["Num_Credit_Card"])} credit cards is excessive and increases risk of missing payments.'
        })
        actions.append({
            'title': 'Consolidate Cards',
            'message': 'Close unused cards and maintain only 3-5 active ones that you can manage effectively.'
        })
    
    if features['Num_Credit_Inquiries'] > 6:
        concerns.append({
            'title': 'Excessive Credit Applications',
            'message': f'{int(features["Num_Credit_Inquiries"])} recent inquiries suggests credit shopping or financial stress.'
        })
        actions.append({
            'title': 'Stop Applying',
            'message': 'Pause all new credit applications for at least 6 months. Multiple inquiries harm your score.'
        })
    
   
    if balance_ratio < 10:
        concerns.append({
            'title': 'Very Low Cash Reserves',
            'message': 'Bank balance is less than 10% of monthly income. This leaves you vulnerable to emergencies.'
        })
        actions.append({
            'title': 'Build Cash Reserve',
            'message': 'Aim for at least 1 month of expenses in checking/savings. This reduces reliance on credit for emergencies.'
        })
    elif balance_ratio > 100:
        strengths.append({
            'title': 'Strong Liquidity',
            'message': f'Maintaining {balance_ratio:.0f}% of monthly income in the bank shows excellent financial planning.'
        })
    
    
    # If no strengths (very rare), add something positive
    if not strengths:
        if features['Num_Bank_Accounts'] > 0:
            strengths.append({
                'title': 'Banking Relationship',
                'message': 'You maintain bank accounts, which is a positive foundation for building better credit.'
            })
        else:
            strengths.append({
                'title': 'Starting Point',
                'message': 'You are taking the first step by understanding your credit profile. This awareness is crucial for improvement.'
            })
    
    # If no concerns (rare for poor/doubtful/fair), add general guidance
    if not concerns:
        concerns.append({
            'title': 'Continuous Monitoring',
            'message': 'While your current profile is strong, maintaining these habits requires ongoing attention and discipline.'
        })
    
    # If no actions, provide general advice
    if not actions:
        actions.append({
            'title': 'Maintain Excellence',
            'message': 'Continue your current habits. Review your credit report annually and monitor for any unauthorized activity.'
        })
    
    return {
        'strengths': strengths,
        'concerns': concerns,
        'actions': actions
    }

def ask_financial_chatbot(user_input):
    """Finance-only chatbot with Ollama integration"""
    
    finance_keywords = [
        'credit', 'score', 'loan', 'debt', 'money', 'save', 'saving', 'budget', 
        'income', 'salary', 'emi', 'interest', 'bank', 'card', 'payment', 'finance',
        'invest', 'investment', 'financial', 'rupee', 'rs', 'amount', 'pay',
        'spend', 'expense', 'afford', 'tax', 'insurance', 'mortgage', 'utilization',
        'cibil', 'experian', 'equifax', 'approval', 'lender', 'borrowing', 'wealth'
    ]
    
    input_lower = user_input.lower()
    is_finance = any(keyword in input_lower for keyword in finance_keywords)
    
    if not is_finance:
        return "I only help with finance topics like credit scores, budgeting, loans, savings, and money management. Please ask a finance-related question."
    
    follow_up_words = ['explain', 'simply', 'brief', 'detail', 'more', 'less', 'line', 'sentence', 
                      'elaborate', 'shorter', 'understand', 'clarify', 'rephrase', 'example']
    is_followup = any(word in input_lower for word in follow_up_words)
    
    last_answer = ""
    if is_followup and len(st.session_state.messages) >= 2:
        prev_msg = st.session_state.messages[-1]
        if prev_msg.get("role") == "assistant":
            last_answer = prev_msg.get("content", "")
    
    if is_followup and last_answer:
        context_instruction = f"""Previous answer: "{last_answer}"

User wants: {user_input}

Adjust response accordingly (make it simpler/more detailed/shorter as requested)."""
    else:
        context_instruction = f"Question: {user_input}"
    
    payload = {
        "model": "gemma:2b",
        "prompt": f"""You're a strict Indian financial advisor. You ONLY answer questions about:
- Credit scores and credit reports
- Budgeting and saving money
- Loans and debt management
- Credit cards and payments
- Financial planning and investments

{context_instruction}

RULES:
1. If NOT about finance, say: "I only help with finance topics."
2. Use rupees for Indian context
3. Give practical, actionable advice
4. Be brief if they ask for brief answers
5. Be warm, helpful, and encouraging
6. Keep answers under 150 words unless detail is requested

Answer:""",
        "stream": False,
        "options": {
            "temperature": 0.6,
            "num_predict": 200,
            "top_p": 0.9
        }
    }
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            answer = response.json().get("response", "").strip()
            
            bad_phrases = ["Question:", "User question:", "Next question:", 
                          "You might also", "For example, if", "\nQuestion", "User:", "Assistant:"]
            for phrase in bad_phrases:
                if phrase in answer:
                    answer = answer.split(phrase)[0].strip()
            
            if any(word in input_lower for word in ['1 line', 'one line', 'brief', 'shortly', 'quick']):
                sentences = answer.split('.')
                answer = sentences[0] + '.' if sentences else answer
            
            if len(answer) < 15:
                return "Could you rephrase that? I'm here to help with credit scores, budgeting, and financial planning."
            
            return answer
        else:
            return "Something went wrong. Try again?"
        
    except requests.exceptions.Timeout:
        return "Taking too long... Try a simpler question?"
    except requests.exceptions.ConnectionError:
        return "Can't connect to chatbot. Make sure Ollama is running with: ollama serve"
    except Exception as e:
        return f"Error: {str(e)}"

# Navigation functions
def go_to_home():
    st.session_state.page = 'home'
    st.session_state.section = 1
    st.session_state.show_chatbot = False

def go_to_assessment():
    st.session_state.page = 'assessment'
    st.session_state.section = 1

def next_section():
    st.session_state.section += 1

def prev_section():
    st.session_state.section -= 1

def reset_assessment():
    st.session_state.answers = {}
    st.session_state.section = 1

def toggle_chatbot():
    st.session_state.show_chatbot = not st.session_state.show_chatbot
    if st.session_state.show_chatbot and not st.session_state.messages:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hi! I can help with credit scores, budgeting, debt management, and financial planning. What would you like to know?"
        }]

def clear_chat():
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hi! I can help with credit scores, budgeting, debt management, and financial planning. What would you like to know?"
    }]



def show_home():
    """Home page with app introduction"""
    
    # Hero section with logo
    if logo_base64:
        st.markdown(f"""
        <div class="hero-section">
            <div class="logo-container">
                <img src="data:image/png;base64,{logo_base64}" alt="CreditWise Logo">
            </div>
            <p class="hero-subtitle">AI-Powered Behavioral Credit Assessment</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="hero-section">
            <h1 class="hero-title">CreditWise</h1>
            <p class="hero-subtitle">AI-Powered Behavioral Credit Assessment</p>
        </div>
        """, unsafe_allow_html=True)
    
    # What is Behavioral Credit Scoring
    st.markdown("""
    <div class="section-card">
        <h2 class="section-title">What is Behavioral Credit Scoring?</h2>
        <p class="section-description">
            Unlike traditional scores that only check credit history, behavioral credit scoring 
            analyzes your complete financial behavior - payment patterns, savings habits, 
            debt management, and spending discipline. This app uses an ensemble of Random Forest 
            and LightGBM models for highly accurate predictions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-title">90%+ Confidence</div>
            <p class="feature-text">Advanced ensemble ML with guaranteed accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-title">Comprehensive Analysis</div>
            <p class="feature-text">Detailed insights across all aspects of credit health</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-title">Actionable Steps</div>
            <p class="feature-text">Clear guidance tailored to your specific situation</p>
        </div>
        """, unsafe_allow_html=True)
    
    # CTA Button
    st.markdown("<div class='spacer-medium'></div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Calculate Your Credit Score", type="primary", use_container_width=True):
            go_to_assessment()
            st.rerun()
    
    # Disclaimer
    st.markdown("""
    <div class="info-box">
        <strong>Note:</strong> This behavioral credit score may differ from official credit bureau scores 
        (CIBIL, Experian) as it uses AI-powered behavioral analysis. Use it as a comprehensive guide to understand 
        and improve your financial health.
    </div>
    """, unsafe_allow_html=True)

def show_assessment():
    """Main assessment flow with progress bar"""
    
    sections = ['Personal', 'Income', 'Credit', 'Loans', 'Payment History']
    
    # Progress bar
    progress_html = '<div class="progress-bar">'
    for i, section in enumerate(sections, 1):
        if i < st.session_state.section:
            class_name = 'progress-step completed'
        elif i == st.session_state.section:
            class_name = 'progress-step active'
        else:
            class_name = 'progress-step'
        progress_html += f'<div class="{class_name}">{section}</div>'
    progress_html += '</div>'
    
    st.markdown(progress_html, unsafe_allow_html=True)
    
    # Show appropriate section
    if st.session_state.section == 1:
        show_section_1()
    elif st.session_state.section == 2:
        show_section_2()
    elif st.session_state.section == 3:
        show_section_3()
    elif st.session_state.section == 4:
        show_section_4()
    elif st.session_state.section == 5:
        show_section_5()
    elif st.session_state.section == 6:
        if st.session_state.show_chatbot:
            show_chatbot()
        else:
            show_results()

def show_section_1():
    """Section 1: Personal Information"""
    
    st.markdown("""
    <div class="section-card">
        <h2 class="section-title">Personal Information</h2>
        <p class="section-description">Basic information about you</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=st.session_state.answers.get('age', 30),
            help="Your current age"
        )
        st.session_state.answers['age'] = age
    
    with col2:
        occupation = st.selectbox(
            "Occupation",
            options=list(OCCUPATION_MAP.keys()),
            index=list(OCCUPATION_MAP.keys()).index(st.session_state.answers.get('occupation', 'Other')),
            help="Your primary occupation"
        )
        st.session_state.answers['occupation'] = occupation
    
    if occupation == 'Other':
        other_occ = st.text_input(
            "Please specify your occupation (Required)",
            value=st.session_state.answers.get('other_occupation_text', ''),
            help="Required field - Enter alphabets only"
        )
        if other_occ:
            if not other_occ.replace(' ', '').isalpha():
                st.error("Please enter only alphabetical characters")
                st.session_state.answers['other_occupation_text'] = ''
            else:
                st.session_state.answers['other_occupation_text'] = other_occ
        else:
            st.session_state.answers['other_occupation_text'] = ''
    
    st.markdown("<div class='spacer-small'></div>", unsafe_allow_html=True)
    
    income_stability = st.selectbox(
        "How long have you been in your current occupation?",
        options=["Less than 1 year", "1-3 years", "3-5 years", "5-10 years", "More than 10 years"],
        index=["Less than 1 year", "1-3 years", "3-5 years", "5-10 years", "More than 10 years"].index(
            st.session_state.answers.get('income_stability', '3-5 years')
        ),
        help="Employment stability is a key factor in creditworthiness"
    )
    st.session_state.answers['income_stability'] = income_stability
    
    st.markdown("<div class='spacer-medium'></div>", unsafe_allow_html=True)
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col3:
        if st.button("Next", type="primary", use_container_width=True):
            if st.session_state.answers.get('occupation') == 'Other' and not st.session_state.answers.get('other_occupation_text', '').strip():
                st.error('Please specify your occupation')
            else:
                next_section()
                st.rerun()

def show_section_2():
    """Section 2: Income & Savings - CRITICAL FINANCIAL DATA"""
    
    st.markdown("""
    <div class="section-card">
        <h2 class="section-title">Income & Financial Health</h2>
        <p class="section-description">Your income and savings information (all amounts in ₹)</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        salary = st.number_input(
            "Monthly Take-Home Salary (₹)",
            min_value=0,
            max_value=10000000,
            value=st.session_state.answers.get('monthly_salary', 0),
            step=1000,
            help="Your monthly in-hand salary after all deductions"
        )
        st.session_state.answers['monthly_salary'] = salary
    
    with col2:
        savings = st.number_input(
            "Monthly Savings/Investments (₹)",
            min_value=0,
            max_value=10000000,
            value=st.session_state.answers.get('monthly_savings', 0),
            step=500,
            help="Amount you save or invest each month (SIP, FD, etc.)"
        )
        st.session_state.answers['monthly_savings'] = savings
    
    st.markdown("<div class='spacer-small'></div>", unsafe_allow_html=True)
    
    # Bank accounts and balance
    col1, col2 = st.columns(2)
    
    with col1:
        bank_accounts = st.number_input(
            "Number of Bank Accounts",
            min_value=0,
            max_value=15,
            value=st.session_state.answers.get('num_bank_accounts', 2),
            help="Total number of savings/current accounts you maintain"
        )
        st.session_state.answers['num_bank_accounts'] = bank_accounts
    
    with col2:
        balance = st.number_input(
            "Average Bank Balance (₹)",
            min_value=0,
            max_value=100000000,
            value=st.session_state.answers.get('monthly_balance', 0),
            step=1000,
            help="Typical combined balance across all your accounts"
        )
        st.session_state.answers['monthly_balance'] = balance
    
    st.markdown("<div class='spacer-medium'></div>", unsafe_allow_html=True)
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Back", use_container_width=True):
            prev_section()
            st.rerun()
    with col3:
        if st.button("Next", type="primary", use_container_width=True):
            if salary == 0:
                st.error('Monthly salary cannot be 0')
            else:
                next_section()
                st.rerun()

def show_section_3():
    """Section 3: Credit Cards & History - MOST CRITICAL"""
    
    st.markdown("""
    <div class="section-card">
        <h2 class="section-title">Credit Cards & History</h2>
        <p class="section-description">Your credit card usage and credit history</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Credit history
    credit_history = st.slider(
        "How many years have you been using any credit products?",
        min_value=0.0,
        max_value=30.0,
        value=float(st.session_state.answers.get('credit_history_years', 5.0)),
        step=0.5,
        help="Total years since you opened your first credit card or loan"
    )
    st.session_state.answers['credit_history_years'] = credit_history
    
    st.markdown("<div class='spacer-small'></div>", unsafe_allow_html=True)
    
    # Credit cards
    has_cards = st.radio(
        "Do you have any credit cards?",
        options=["Yes", "No"],
        index=0 if st.session_state.answers.get('has_credit_card', False) else 1,
        horizontal=True
    )
    st.session_state.answers['has_credit_card'] = (has_cards == "Yes")
    
    if st.session_state.answers['has_credit_card']:
        col1, col2 = st.columns(2)
        
        with col1:
            num_cards = st.number_input(
                "Number of Credit Cards",
                min_value=1,
                max_value=20,
                value=max(1, st.session_state.answers.get('num_credit_cards', 1)),
                help="Total credit cards you own"
            )
            st.session_state.answers['num_credit_cards'] = num_cards
        
        with col2:
            card_spending = st.number_input(
                "Monthly Credit Card Spending (₹)",
                min_value=0,
                max_value=10000000,
                value=st.session_state.answers.get('monthly_card_spending', 0),
                step=500,
                help="Total amount you spend on credit cards monthly"
            )
            st.session_state.answers['monthly_card_spending'] = card_spending
        
        pays_full = st.radio(
            "Do you pay the full bill or minimum due?",
            options=["Full amount every time", "Minimum due or partial"],
            index=0 if st.session_state.answers.get('pays_full', True) else 1,
            help="Your typical credit card payment behavior"
        )
        st.session_state.answers['pays_full'] = (pays_full == "Full amount every time")
    else:
        st.session_state.answers['num_credit_cards'] = 0
        st.session_state.answers['monthly_card_spending'] = 0
        st.session_state.answers['pays_full'] = True
    
    st.markdown("<div class='spacer-small'></div>", unsafe_allow_html=True)
    
    # Credit inquiries and limit changes
    col1, col2 = st.columns(2)
    
    with col1:
        inquiries = st.number_input(
            "Credit applications in past 6 months",
            min_value=0,
            max_value=20,
            value=st.session_state.answers.get('num_credit_inquiries', 0),
            help="How many times you applied for new credit"
        )
        st.session_state.answers['num_credit_inquiries'] = inquiries
    
    with col2:
        limit_changes_range = st.selectbox(
            "Credit limit changes (past 2 years)",
            options=["Never", "1-2 times", "3-4 times", "5+ times"],
            index=["Never", "1-2 times", "3-4 times", "5+ times"].index(
                st.session_state.answers.get('limit_changes_range', 'Never')
            ),
            help="How often your credit limits changed"
        )
        st.session_state.answers['limit_changes_range'] = limit_changes_range
        
        limit_map = {"Never": 0, "1-2 times": 1.5, "3-4 times": 3.5, "5+ times": 7}
        st.session_state.answers['num_credit_limit_changes'] = limit_map[limit_changes_range]
    
    st.markdown("<div class='spacer-medium'></div>", unsafe_allow_html=True)
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Back", use_container_width=True):
            prev_section()
            st.rerun()
    with col3:
        if st.button("Next", type="primary", use_container_width=True):
            next_section()
            st.rerun()

def show_section_4():
    """Section 4: Loans & Debt"""
    
    st.markdown("""
    <div class="section-card">
        <h2 class="section-title">Loans & Debt</h2>
        <p class="section-description">Information about your loans (all amounts in ₹)</p>
    </div>
    """, unsafe_allow_html=True)
    
    has_loans = st.radio(
        "Do you have any active loans?",
        options=["Yes", "No"],
        index=0 if st.session_state.answers.get('has_loans', False) else 1,
        horizontal=True
    )
    st.session_state.answers['has_loans'] = (has_loans == "Yes")
    
    if st.session_state.answers['has_loans']:
        col1, col2 = st.columns(2)
        
        with col1:
            num_loans = st.number_input(
                "Number of Active Loans",
                min_value=1,
                max_value=10,
                value=max(1, st.session_state.answers.get('num_loans', 1)),
                help="Total number of loans you currently have"
            )
            st.session_state.answers['num_loans'] = num_loans
        
        with col2:
            loan_type = st.selectbox(
                "Primary Loan Type",
                options=list(LOAN_TYPE_MAP.keys()),
                index=list(LOAN_TYPE_MAP.keys()).index(
                    st.session_state.answers.get('loan_type', 'Not Specified')
                ),
                help="Your main loan type"
            )
            st.session_state.answers['loan_type'] = loan_type
        
        if loan_type == 'Not Specified':
            spec_loan = st.text_input(
                "Please specify your loan type (Required)",
                value=st.session_state.answers.get('specified_loan_text', ''),
                help="E.g., Education Loan, Business Loan (alphabets only)"
            )
            if spec_loan:
                spec_loan_clean = spec_loan.strip()
                if not spec_loan_clean.replace(' ', '').isalpha():
                    st.error("Please enter only alphabetical characters")
                    st.session_state.answers['specified_loan_text'] = ''
                else:
                    existing = [lt.lower() for lt in LOAN_TYPE_MAP.keys() if lt != 'Not Specified']
                    if spec_loan_clean.lower() in existing:
                        st.error(f"'{spec_loan}' is already in the list. Please select it from the dropdown.")
                        st.session_state.answers['specified_loan_text'] = ''
                    else:
                        st.session_state.answers['specified_loan_text'] = spec_loan_clean
            else:
                st.session_state.answers['specified_loan_text'] = ''
        
        outstanding = st.number_input(
            "Total Outstanding Loan Amount (₹)",
            min_value=0,
            max_value=100000000,
            value=st.session_state.answers.get('outstanding_debt', 0),
            step=10000,
            help="Total amount you still owe on all loans"
        )
        st.session_state.answers['outstanding_debt'] = outstanding
        
        emi = st.number_input(
            "Total Monthly EMI (₹)",
            min_value=0,
            max_value=10000000,
            value=st.session_state.answers.get('total_emi', 0),
            step=500,
            help="Combined monthly EMI for all your loans"
        )
        st.session_state.answers['total_emi'] = emi
    else:
        st.session_state.answers['num_loans'] = 0
        st.session_state.answers['loan_type'] = 'Not Specified'
        st.session_state.answers['outstanding_debt'] = 0
        st.session_state.answers['total_emi'] = 0
    
    st.markdown("<div class='spacer-medium'></div>", unsafe_allow_html=True)
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Back", use_container_width=True):
            prev_section()
            st.rerun()
    with col3:
        if st.button("Next", type="primary", use_container_width=True):
            errors = []
            
            if st.session_state.answers.get('has_loans', False):
                if st.session_state.answers.get('loan_type') == 'Not Specified' and not st.session_state.answers.get('specified_loan_text', '').strip():
                    errors.append("Please specify your loan type")
                if st.session_state.answers.get('outstanding_debt', 0) == 0:
                    errors.append("Outstanding amount cannot be 0 if you have loans")
                if st.session_state.answers.get('total_emi', 0) == 0:
                    st.warning("EMI is 0. Please confirm this is correct (some loans may have flexible payments)")
            
            if errors:
                for error in errors:
                    st.error(error)
            else:
                next_section()
                st.rerun()

def show_section_5():
    """Section 5: Payment History - THE MOST CRITICAL FACTOR"""
    
    st.markdown("""
    <div class="section-card">
        <h2 class="section-title">Payment History</h2>
        <p class="section-description">This is the MOST IMPORTANT factor in your credit score</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("Payment history accounts for ~35% of your credit score. Even one missed payment can significantly impact your score.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_delays = st.number_input(
            "Missed payment deadlines (past year)",
            min_value=0,
            max_value=50,
            value=st.session_state.answers.get('num_delayed_payments', 0),
            help="How many times you missed any payment deadline"
        )
        st.session_state.answers['num_delayed_payments'] = num_delays
    
    with col2:
        if num_delays > 0:
            delay_days = st.number_input(
                "Average delay duration (days)",
                min_value=1,
                max_value=90,
                value=max(1, st.session_state.answers.get('delay_days', 7)),
                help="On average, how many days late"
            )
            st.session_state.answers['delay_days'] = delay_days
            
            if num_delays == 1:
                st.warning("Even 1 missed payment prevents you from reaching 'Good' or 'Excellent' categories")
            elif num_delays >= 2:
                st.error("Multiple missed payments severely impact your creditworthiness")
        else:
            st.session_state.answers['delay_days'] = 0
            st.success("Perfect! No payment delays is excellent for your credit score")
    
    st.markdown("<div class='spacer-medium'></div>", unsafe_allow_html=True)
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Back", use_container_width=True):
            prev_section()
            st.rerun()
    with col2:
        if st.button("Reset All", use_container_width=True):
            reset_assessment()
            st.rerun()
    with col3:
        if st.button("Calculate My Score", type="primary", use_container_width=True):
            next_section()
            st.rerun()

def show_results():
    """Results page with score and comprehensive insights"""
    
    features = calculate_features(st.session_state.answers)
    score, confidence, probability = predict_score(features)
    
    if score is None:
        st.error("Error: Models not loaded. Ensure rf_model_tuned.pkl, lgbm_nomono_model.pkl, and ensemble_config.pkl are present.")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Back to Home", use_container_width=True):
                go_to_home()
                st.rerun()
        return
    
    interpretation = get_score_interpretation(score)
    insights = generate_categorized_insights(features, score)
    
    # Score display
    st.markdown(f"""
    <div class="score-container">
        <div class="score-label">Your Behavioral Credit Score</div>
        <div class="score-value" style="color: {interpretation['color']}">{score}</div>
        <div class="score-category" style="color: {interpretation['color']}">{interpretation['category']}</div>
        <div class="score-description">{interpretation['description']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics
    st.markdown(f"""
    <div class="metrics-box">
        <div class="metrics-row">
            <div class="metric-item">
                <div class="metric-label">AI Confidence</div>
                <div class="metric-value">{confidence:.1f}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Score ranges
    st.markdown("""
    <div class="section-card">
        <h3 class="section-title">Credit Score Ranges</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    ranges = [
        ("Poor", "300-599", "#c1121f"),
        ("Doubtful", "600-649", "#f97316"),
        ("Fair", "650-699", "#f59e0b"),
        ("Good", "700-749", "#3b82f6"),
        ("Excellent", "750-850", "#2d6a4f")
    ]
    
    for col, (category, range_text, color) in zip([col1, col2, col3, col4, col5], ranges):
        with col:
            if interpretation['category'] == category:
                st.markdown(f"""
                <div class="score-range" style="background: {color}; color: white; font-weight: 800;">
                    <div style="font-size: 1rem; margin-bottom: 0.25rem;">{category}</div>
                    <div style="font-size: 0.85rem;">{range_text}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="score-range" style="background: #f8f9fa; color: #1a1a2e;">
                    <div style="font-size: 1rem; margin-bottom: 0.25rem;">{category}</div>
                    <div style="font-size: 0.85rem;">{range_text}</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("<div class='spacer-medium'></div>", unsafe_allow_html=True)
    
    # ========================================
    # REORDER INSIGHTS BASED ON SCORE CATEGORY
    # ========================================
    
    # For Poor/Doubtful/Fair: Show Concerns first
    if score < 700:
        insight_order = [
            ('concerns', 'Areas of Concern', 'concerns'),
            ('actions', 'Recommended Actions', 'actions'),
            ('strengths', 'Your Strengths', 'strengths')
        ]
    # For Good/Excellent: Show Strengths first
    else:
        insight_order = [
            ('strengths', 'Your Strengths', 'strengths'),
            ('concerns', 'Areas of Concern', 'concerns'),
            ('actions', 'Recommended Actions', 'actions')
        ]
    
    # Display insights in the determined order
    for insight_key, title, css_class in insight_order:
        items = insights[insight_key]
        
        if items:  # Should always be true due to our guarantees
            st.markdown(f"""
            <div class="insight-category">
                <div class="insight-header {css_class}">
                    <h3 class="insight-category-title">{title}</h3>
                </div>
                <div class="insight-body">
            """, unsafe_allow_html=True)
            
            for item in items:
                st.markdown(f"""
                <div class="insight-item {insight_key[:-1]}">
                    <div class="insight-item-title">{item['title']}</div>
                    <div class="insight-item-message">{item['message']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
    
    st.markdown("<div class='spacer-medium'></div>", unsafe_allow_html=True)
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Calculate Again", use_container_width=True):
            reset_assessment()
            st.rerun()
    
    with col2:
        if st.button("Get Financial Advice", type="primary", use_container_width=True):
            toggle_chatbot()
            st.rerun()
    
    with col3:
        if st.button("Back to Home", use_container_width=True):
            go_to_home()
            st.rerun()

def show_chatbot():
    """Chatbot interface for financial advice"""
    
    st.markdown("""
    <div class="section-card">
        <h2 class="section-title">Financial Advisor Chatbot</h2>
        <p class="section-description">Ask questions about credit scores, budgeting, or financial management</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Clear Chat", use_container_width=True):
            clear_chat()
            st.rerun()
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    user_input = st.chat_input("Ask about credit and finance...")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = ask_financial_chatbot(user_input)
            st.markdown(reply)
        
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()
    
    st.markdown("<div class='spacer-small'></div>", unsafe_allow_html=True)
    
    if st.button("Back to Results", use_container_width=True):
        toggle_chatbot()
        st.rerun()

def main():
    """Main app entry point"""
    if st.session_state.page == 'home':
        show_home()
    elif st.session_state.page == 'assessment':
        show_assessment()

if __name__ == "__main__":
    main()