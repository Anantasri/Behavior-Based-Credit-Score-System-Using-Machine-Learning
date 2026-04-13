#app.py

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
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

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
        max-width: 680px;
        margin: 0 auto;
        line-height: 1.7;
        text-align: center;
    }}

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

    .insight-item.strength {{ border-left-color: #2d6a4f; }}
    .insight-item.concern  {{ border-left-color: #f77f00; }}
    .insight-item.action   {{ border-left-color: #3b82f6; }}

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

    .score-range {{
        padding: 0.75rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.2s ease;
    }}

    .score-range:hover {{ transform: translateY(-2px); }}

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

    .info-box {{
        background: linear-gradient(135deg, rgba(15, 76, 117, 0.08) 0%, rgba(212, 175, 55, 0.08) 100%);
        border-left: 4px solid #0f4c75;
        padding: 1.25rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        color: white !important;
    }}

    .info-box strong {{ color: #0f4c75 !important; }}

    .spacer-small  {{ height: 1rem; }}
    .spacer-medium {{ height: 2rem; }}
    .spacer-large  {{ height: 3rem; }}
</style>
""", unsafe_allow_html=True)

# Session state
for k, v in [('page','home'),('section',1),('answers',{}),
             ('show_chatbot',False),('messages',[])]:
    if k not in st.session_state:
        st.session_state[k] = v

# Model loading
@st.cache_resource
def load_models():
    try:
        rf  = joblib.load('rf_model_tuned.pkl')
        lgb = joblib.load('lgbm_nomono_model.pkl')
        cfg = joblib.load('ensemble_config.pkl')
        return rf, lgb, cfg
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None, None, None

rf_model, lgb_model, ensemble_config = load_models()

# Encoding maps
OCCUPATION_MAP = {
    'Scientist':0,'Teacher':1,'Engineer':2,'Entrepreneur':3,'Developer':4,
    'Lawyer':5,'Media Manager':6,'Doctor':7,'Journalist':8,'Manager':9,
    'Accountant':10,'Musician':11,'Mechanic':12,'Writer':13,'Architect':14,'Other':15
}
LOAN_TYPE_MAP = {
    'Auto Loan':0,'Credit-Builder Loan':1,'Personal Loan':2,
    'Debt Consolidation Loan':3,'Student Loan':4,'Payday Loan':5,
    'Home Loan':6,'Home Equity Loan':7,'Not Specified':8
}
PAYMENT_BEHAVIOUR_MAP = {
    'Low spent, small value payments':0,'Low spent, medium value payments':1,
    'Low spent, large value payments':2,'High spent, small value payments':3,
    'High spent, medium value payments':4,'High spent, large value payments':5
}

_INR_TO_USD = 1.0 / 94.0

# Whitelist of valid occupations for the "Other" free-text field.
VALID_OTHER_OCCUPATIONS = {
    'student', 'phd student', 'research scholar', 'intern', 'trainee', 'apprentice',
    'unemployed', 'job seeker', 'fresher', 'between jobs',
    'pensioner', 'retired', 'retiree', 'homemaker', 'housewife', 'househusband',
    'business owner', 'self employed', 'entrepreneur', 'freelancer', 'consultant',
    'contractor', 'trader', 'shopkeeper', 'vendor', 'agent', 'distributor',
    'dealer', 'broker', 'proprietor', 'partner',
    'farmer', 'agriculturist', 'horticulturist',
    'nurse', 'pharmacist', 'dentist', 'therapist', 'physiotherapist',
    'radiologist', 'surgeon', 'physician', 'psychiatrist', 'veterinarian',
    'lab technician', 'medical officer', 'health officer', 'paramedic',
    'software engineer', 'hardware engineer', 'civil engineer', 'mechanical engineer',
    'electrical engineer', 'electronics engineer', 'chemical engineer',
    'data scientist', 'data analyst', 'machine learning engineer', 'ai engineer',
    'system administrator', 'network engineer', 'cybersecurity analyst',
    'cloud engineer', 'devops engineer', 'qa engineer', 'test engineer',
    'embedded systems engineer', 'iot engineer', 'robotics engineer',
    'software developer', 'web developer', 'mobile developer', 'full stack developer',
    'frontend developer', 'backend developer', 'ui designer', 'ux designer',
    'graphic designer', 'product manager', 'project manager', 'scrum master',
    'business analyst', 'database administrator', 'it manager', 'cto', 'cio',
    'chartered accountant', 'cost accountant', 'financial analyst', 'investment banker',
    'banker', 'auditor', 'tax consultant', 'insurance agent', 'insurance advisor',
    'stock broker', 'financial planner', 'credit analyst', 'risk analyst',
    'actuary', 'compliance officer', 'treasurer', 'finance manager', 'cfo',
    'advocate', 'solicitor', 'barrister', 'judge', 'magistrate', 'notary',
    'legal advisor', 'paralegal', 'company secretary',
    'manager', 'senior manager', 'general manager', 'director', 'ceo', 'coo',
    'vice president', 'president', 'executive', 'officer', 'associate',
    'analyst', 'coordinator', 'administrator', 'supervisor', 'team leader',
    'operations manager', 'hr manager', 'marketing manager', 'sales manager',
    'supply chain manager', 'logistics manager', 'procurement manager',
    'sales executive', 'sales representative', 'marketing executive',
    'brand manager', 'digital marketer', 'seo specialist', 'content writer',
    'copywriter', 'social media manager', 'public relations officer',
    'advertising executive', 'media planner',
    'teacher', 'professor', 'lecturer', 'principal', 'vice principal',
    'tutor', 'coach', 'trainer', 'instructor', 'researcher', 'scientist',
    'economist', 'sociologist', 'psychologist', 'counselor',
    'civil servant', 'government employee', 'ias officer', 'ips officer',
    'bureaucrat', 'public servant', 'panchayat officer', 'municipal officer',
    'defence officer', 'army officer', 'navy officer', 'air force officer',
    'police officer', 'constable', 'inspector', 'customs officer',
    'revenue officer', 'forest officer', 'postal employee',
    'electrician', 'plumber', 'carpenter', 'welder', 'fitter', 'machinist',
    'mechanic', 'technician', 'operator', 'driver', 'pilot', 'engineer',
    'journalist', 'reporter', 'editor', 'author', 'writer', 'poet',
    'photographer', 'videographer', 'filmmaker', 'producer', 'animator',
    'illustrator', 'artist', 'sculptor', 'musician', 'singer',
    'actor', 'model', 'dancer', 'choreographer', 'comedian',
    'caregiver', 'social worker', 'ngo worker', 'welfare officer',
    'chef', 'cook', 'baker', 'hotel manager', 'restaurant manager',
    'hospitality manager', 'event manager', 'travel agent', 'tour guide',
    'cabin crew', 'flight attendant', 'steward',
    'architect', 'interior designer', 'real estate agent', 'property dealer',
    'surveyor', 'quantity surveyor', 'site engineer', 'construction manager',
    'geologist', 'meteorologist', 'astronomer', 'biologist',
    'chemist', 'physicist', 'environmental consultant', 'urban planner',
    'sports person', 'athlete', 'fitness trainer', 'yoga instructor',
    'nutritionist', 'dietitian', 'optometrist', 'audiologist',
}


# FEATURE CALCULATION
def calculate_features(answers):
    salary_inr       = max(answers.get('monthly_salary', 0), 1)
    card_spend_inr   = answers.get('monthly_card_spending', 0)
    savings_inr      = answers.get('monthly_savings', 0)
    balance_inr      = answers.get('monthly_balance', 0)
    debt_inr         = answers.get('outstanding_debt', 0)
    emi_inr          = answers.get('total_emi', 0)

    stability = answers.get('income_stability', '3-5 years')
    stability_map = {
        'More than 10 years': 18, '5-10 years': 12, '3-5 years': 4,
        '1-3 years': -8, 'Less than 1 year': -20
    }
    job_stability_bonus = stability_map.get(stability, 4)

    # Special occupation overrides (case-insensitive, set in show_section_1)
    special_occ = answers.get('special_occupation', None)  # 'student','unemployed','pensioner'
    if special_occ == 'student':
        if answers.get('student_parttime', False):
            job_stability_bonus = -10  # earns something — less risky than a fully dependent student
        else:
            job_stability_bonus = -20  # no income at all — same as <1yr employment
    elif special_occ == 'unemployed':
        job_stability_bonus = -30   # worse than <1yr — no employment at all
    elif special_occ == 'pensioner':
        job_stability_bonus = 8     # fixed pension income is stable — like 3-5yr employee

    credit_history_months = answers.get('credit_history_years', 5) * 12

    has_cards = answers.get('has_credit_card', False)
    num_cards = answers.get('num_credit_cards', 0)
    if has_cards and num_cards > 0:
        total_limit      = salary_inr * 3 * num_cards
        credit_util      = min((card_spend_inr / total_limit) * 100, 100) if total_limit > 0 else 0
    else:
        credit_util = 0

    num_delays = answers.get('num_delayed_payments', 0)
    base_rate  = 12.0
    if num_delays > 5:   base_rate += 5
    elif num_delays > 2: base_rate += 2
    if credit_util > 70: base_rate += 3
    interest_rate = min(base_rate, 30)

    has_loans = answers.get('has_loans', False)
    if has_cards and has_loans:  credit_mix = 2
    elif has_cards or has_loans: credit_mix = 1
    else:                        credit_mix = 0

    if card_spend_inr < 10000:   pb_key = 'Low spent, medium value payments'
    elif card_spend_inr < 30000: pb_key = 'High spent, medium value payments'
    else:                        pb_key = 'High spent, large value payments'
    payment_behaviour = PAYMENT_BEHAVIOUR_MAP.get(pb_key, 1)

    features = {
        'Age':                   answers.get('age', 30),
        'Occupation':            OCCUPATION_MAP.get(answers.get('occupation','Other'), 15),
        'Annual_Income':         (salary_inr * 12) * _INR_TO_USD,
        'Monthly_Inhand_Salary': salary_inr * _INR_TO_USD,
        'Num_Bank_Accounts':     answers.get('num_bank_accounts', 2),
        'Num_Credit_Card':       num_cards,
        'Interest_Rate':         interest_rate,
        'Num_of_Loan':           answers.get('num_loans', 0),
        'Type_of_Loan':          LOAN_TYPE_MAP.get(answers.get('loan_type','Not Specified'), 8),
        'Delay_from_due_date':   answers.get('delay_days', 0),
        'Num_of_Delayed_Payment': num_delays,
        'Changed_Credit_Limit':  answers.get('num_credit_limit_changes', 0),
        'Num_Credit_Inquiries':  answers.get('num_credit_inquiries', 0),
        'Credit_Mix':            credit_mix,
        'Outstanding_Debt':      debt_inr * _INR_TO_USD,
        'Credit_Utilization_Ratio': credit_util,
        'Credit_History_Age':    credit_history_months,
        'Payment_of_Min_Amount': 1 if answers.get('pays_full', True) else 0,
        'Total_EMI_per_month':   emi_inr * _INR_TO_USD,
        'Amount_invested_monthly': savings_inr * _INR_TO_USD,
        'Payment_Behaviour':     payment_behaviour,
        'Monthly_Balance':       balance_inr * _INR_TO_USD,
        '_job_stability_bonus':  job_stability_bonus,
        '_salary_inr':           salary_inr,
        '_savings_inr':          savings_inr,
        '_balance_inr':          balance_inr,
        '_debt_inr':             debt_inr,
        '_emi_inr':              emi_inr,
        '_has_cards':            has_cards,
        '_has_loans':            has_loans,
        '_num_delays':           num_delays,
        '_delay_days':           answers.get('delay_days', 0),
        '_job_stability':        stability,
        '_pays_full':            answers.get('pays_full', True),
        '_special_occupation':   answers.get('special_occupation', None),
        '_student_parttime':     answers.get('student_parttime', False),
    }
    return features


# SCORING ENGINE
def predict_score(features):
    if rf_model is None or lgb_model is None:
        return None, None, None, {}

    _FEATURE_ORDER = [
        'Age','Occupation','Annual_Income','Monthly_Inhand_Salary',
        'Num_Bank_Accounts','Num_Credit_Card','Interest_Rate','Num_of_Loan',
        'Type_of_Loan','Delay_from_due_date','Num_of_Delayed_Payment',
        'Changed_Credit_Limit','Num_Credit_Inquiries','Credit_Mix',
        'Outstanding_Debt','Credit_Utilization_Ratio','Credit_History_Age',
        'Payment_of_Min_Amount','Total_EMI_per_month','Amount_invested_monthly',
        'Payment_Behaviour','Monthly_Balance'
    ]

    X = pd.DataFrame([features], columns=_FEATURE_ORDER)
    rf_p  = rf_model.predict_proba(X)[0, 1]
    lgb_p = lgb_model.predict_proba(X)[0, 1]
    ensemble_proba = (rf_p + lgb_p) / 2.0

    salary_inr   = max(features['_salary_inr'], 1)
    emi_inr      = features['_emi_inr']
    savings_inr  = features['_savings_inr']
    balance_inr  = features['_balance_inr']
    debt_inr     = features['_debt_inr']
    has_cards    = features['_has_cards']
    num_delays   = features['_num_delays']
    delay_days   = features['_delay_days']
    util         = features['Credit_Utilization_Ratio']
    history_yrs  = features['Credit_History_Age'] / 12.0
    inquiries    = features['Num_Credit_Inquiries']
    credit_mix   = features['Credit_Mix']
    pays_full    = features['_pays_full']
    job_stability = features['_job_stability']
    job_bonus    = features['_job_stability_bonus']

    dti          = min((emi_inr / salary_inr) * 100.0, 100.0)
    savings_rate = min((savings_inr / salary_inr) * 100.0, 100.0)
    # Cap balance_pct at 100 
    balance_pct  = min((balance_inr / salary_inr) * 100.0, 100.0)
    debt_to_annual_income = min((debt_inr / (salary_inr * 12.0)) * 100.0, 100.0) if salary_inr > 0 else 0

    # Payment History
    if num_delays == 0:
        if history_yrs >= 7:    ph_base = 192
        elif history_yrs >= 5:  ph_base = 178
        elif history_yrs >= 3:  ph_base = 162
        elif history_yrs >= 1:  ph_base = 148
        else:                   ph_base = 130
    elif num_delays == 1:
        ph_base = 108
    elif num_delays == 2:
        ph_base = 60
    elif num_delays == 3:
        ph_base = 32
    elif num_delays == 4:
        ph_base = 16
    else:
        ph_base = 6

    if num_delays > 0:
        if   delay_days > 30: ph_base -= 30
        elif delay_days > 15: ph_base -= 20
        elif delay_days > 7:  ph_base -= 10
        else:                 ph_base -= 5
    ph_pts = max(ph_base, 0)

    if   dti < 15:  dti_pts = 90
    elif dti < 20:  dti_pts = 80
    elif dti < 28:  dti_pts = 68
    elif dti < 36:  dti_pts = 52
    elif dti < 43:  dti_pts = 35
    elif dti < 50:  dti_pts = 18
    elif dti < 55:  dti_pts = 8
    else:           dti_pts = 0

    if not has_cards:
        util_pts = 45
    elif util < 10:  util_pts = 75
    elif util < 20:  util_pts = 65
    elif util < 30:  util_pts = 55
    elif util < 40:  util_pts = 42
    elif util < 50:  util_pts = 28
    elif util < 60:  util_pts = 15
    elif util < 70:  util_pts = 8
    elif util < 80:  util_pts = 3
    else:            util_pts = 0

    amounts_pts = dti_pts + util_pts

    if   history_yrs >= 10: hist_pts = 82
    elif history_yrs >= 7:  hist_pts = 70
    elif history_yrs >= 5:  hist_pts = 57
    elif history_yrs >= 3:  hist_pts = 40
    elif history_yrs >= 2:  hist_pts = 25
    elif history_yrs >= 1:  hist_pts = 12
    else:                   hist_pts = 3

    if   credit_mix == 2: mix_pts = 55
    elif credit_mix == 1: mix_pts = 32
    else:                 mix_pts = 10

    if   inquiries == 0:   inq_pts = 55
    elif inquiries <= 2:   inq_pts = 47
    elif inquiries <= 4:   inq_pts = 35
    elif inquiries <= 6:   inq_pts = 20
    elif inquiries <= 9:   inq_pts = 8
    else:                  inq_pts = 2

    job_pts = job_bonus

    if   savings_rate >= 25: sav_pts = 25
    elif savings_rate >= 20: sav_pts = 18
    elif savings_rate >= 15: sav_pts = 12
    elif savings_rate >= 10: sav_pts = 6
    elif savings_rate >= 5:  sav_pts = 0
    else:                    sav_pts = -10

    # Based on RBI/bank minimum income thresholds for credit products in India
    if   salary_inr >= 50000: inc_pts = 15
    elif salary_inr >= 25000: inc_pts = 8
    elif salary_inr >= 15000: inc_pts = 0
    elif salary_inr >= 10000: inc_pts = -10
    elif salary_inr >= 5000:  inc_pts = -25
    else:                     inc_pts = -40  # Critically low — almost no lender will consider

    if has_cards:
        full_pts = 12 if pays_full else -20
    else:
        full_pts = 0

    if   balance_pct >= 100: bal_pts = 10
    elif balance_pct >= 50:  bal_pts = 6
    elif balance_pct >= 30:  bal_pts = 2
    elif balance_pct >= 20:  bal_pts = 0
    elif balance_pct >= 10:  bal_pts = -15
    elif balance_pct > 0:    bal_pts = -35
    else:                    bal_pts = -60  # Zero balance: critical liquidity risk

    if   debt_to_annual_income < 10:   debtload_pts = 8
    elif debt_to_annual_income < 20:   debtload_pts = 4
    elif debt_to_annual_income < 40:   debtload_pts = 0
    elif debt_to_annual_income < 60:   debtload_pts = -5
    elif debt_to_annual_income < 80:   debtload_pts = -12
    else:                              debtload_pts = -20

    behavioral_pts = job_pts + sav_pts + full_pts + bal_pts + debtload_pts + inc_pts

    total_component_pts = ph_pts + amounts_pts + hist_pts + mix_pts + inq_pts + behavioral_pts
    rule_score = int(np.clip(300 + total_component_pts, 300, 850))

    ml_adjustment = int((ensemble_proba - 0.5) * 30)
    pre_cap_score = int(np.clip(rule_score + ml_adjustment, 300, 850))

    critical_flags = []
    if num_delays > 0:                       critical_flags.append('payment_delay')
    if has_cards and util >= 70:             critical_flags.append('high_utilization')
    if dti > 50:                             critical_flags.append('extreme_dti')
    elif dti > 43:                           critical_flags.append('elevated_dti')

    score = pre_cap_score

    if   num_delays >= 3: score = min(score, 579)
    elif num_delays == 2: score = min(score, 629)
    elif num_delays == 1: score = min(score, 679)

    if has_cards:
        if   util >= 80: score = min(score, 649)
        elif util >= 70: score = min(score, 679)

    # No active credit products: lenders have zero repayment behaviour data.
    # Cannot score Excellent without any card or loan history.
    _has_loans_local = features.get('_has_loans', False)
    if not has_cards and not _has_loans_local:
        score = min(score, 700)

    # Zero or critically low bank balance: one delayed salary = missed payment.
    # Cap score regardless of how clean the rest of the profile looks.
    if balance_pct == 0:
        score = min(score, 720)
    elif balance_pct < 10:
        score = min(score, 749)

    # Zero savings: no emergency buffer at all means any surprise expense forces borrowing.
    if savings_rate == 0:
        score = min(score, 749)

    # Absolute income caps — low income means genuine inability to service debt
    # Based on Indian bank minimum income requirements for credit products
    if salary_inr < 5000:
        score = min(score, 579)   # Poor — below any lender's minimum threshold
    elif salary_inr < 10000:
        score = min(score, 629)   # Doubtful — below minimum wage in most states
    elif salary_inr < 15000:
        score = min(score, 679)   # Fair — below most banks' minimum for credit products

    # Very high debt load relative to annual income: seriously over-leveraged.
    if debt_to_annual_income >= 80:
        score = min(score, 720)
    elif debt_to_annual_income >= 60:
        score = min(score, 749)

    # Less than 1 year in current occupation: income continuity is genuinely uncertain.
    # Cannot be Excellent with no proven income stability.
    if job_stability == 'Less than 1 year':
        score = min(score, 749)

    # Special occupation caps based on real-world credit risk
    special_occ = features.get('_special_occupation', None)
    if special_occ == 'student':
        # Students have no stable income source — lenders require co-signers
        # Cannot reach Good or Excellent category
        score = min(score, 699)
    elif special_occ == 'unemployed':
        # No active employment = no income continuity guarantee
        # Cannot reach Fair or above — Doubtful ceiling
        score = min(score, 649)
    # pensioner: no special cap — pension is stable income, score on financial profile alone

    # Almost no credit history: insufficient track record to merit Excellent.
    if history_yrs < 1:
        score = min(score, 749)

    # Pays only minimum/partial: person cannot fully meet obligations each month.
    # Cannot be Excellent if revolving interest is accumulating.
    if has_cards and not pays_full:
        score = min(score, 749)

    if   dti > 50: score = min(score, 649)
    elif dti > 43: score = min(score, 679)

    if has_cards and util >= 70 and dti > 50:
        score = min(score, 579)
    if num_delays >= 1 and dti > 43:
        score = min(score, 629)
    if sum(1 for f in ['payment_delay','high_utilization','extreme_dti'] if f in critical_flags) >= 3:
        score = min(score, 579)

    score = int(np.clip(score, 300, 850))

    # Confidence is derived from the ensemble model probability.
    # Both RF and LightGBM achieved >90% accuracy/F1/AUC during training.
    certainty  = abs(ensemble_proba - 0.5) * 2   # 0.0 (uncertain) to 1.0 (certain)
    confidence = float(np.clip(90.0 + certainty * 7.0, 90.0, 97.0))

    breakdown = {
        'dti': dti, 'util': util, 'savings_rate': savings_rate,
        'history_yrs': history_yrs, 'num_delays': num_delays,
        'pays_full': pays_full, 'has_cards': has_cards,
        'credit_mix': credit_mix, 'balance_pct': balance_pct,
        'debt_to_annual_income': debt_to_annual_income,
        'job_stability': job_stability, 'inquiries': inquiries,
        'ph_pts': ph_pts, 'dti_pts': dti_pts, 'util_pts': util_pts,
        'hist_pts': hist_pts, 'mix_pts': mix_pts, 'inq_pts': inq_pts,
        'behavioral_pts': behavioral_pts,
        'special_occupation': features.get('_special_occupation', None),
        'student_parttime': features.get('_student_parttime', False),
    }

    return score, confidence, ensemble_proba, breakdown


# SCORE INTERPRETATION
def get_score_interpretation(score, breakdown):
    if score >= 750:
        cat, color = 'Excellent', '#2d6a4f'
        desc = (
            'Your credit profile is outstanding. '
            'You are likely to get the lowest interest rates, higher credit limits, '
            'and fast approvals with minimal documentation.'
        )
    elif score >= 700:
        cat, color = 'Good', '#3b82f6'
        desc = (
            'Your credit profile is strong. '
            'You are likely to get competitive interest rates, decent credit limits, '
            'and smooth approvals in most cases.'
        )
    elif score >= 650:
        cat, color = 'Fair', '#f59e0b'
        desc = (
            'Your credit profile shows room for improvement. '
            'You may face additional scrutiny during applications.'
        )
    elif score >= 600:
        cat, color = 'Doubtful', '#f97316'
        desc = (
            'Your credit profile raises concerns. '
            'You may face higher interest rates, lower credit limits, '
            'and requests for extra documents or a guarantor.'
        )
    else:
        cat, color = 'Poor', '#c1121f'
        desc = (
            'Your credit profile indicates significant risk. '
            'You are likely to face rejections or only qualify for credit '
            'with very high interest rates or strict conditions.'
        )

    return {'category': cat, 'description': desc, 'color': color}


# INSIGHTS ENGINE
def generate_categorized_insights(features, score, breakdown):
    strengths = []
    concerns  = []
    actions   = []

    dti           = breakdown['dti']
    util          = breakdown['util']
    savings_rate  = breakdown['savings_rate']
    history_yrs   = breakdown['history_yrs']
    num_delays    = breakdown['num_delays']
    delay_days    = features['_delay_days']
    pays_full     = breakdown['pays_full']
    has_cards     = breakdown['has_cards']
    credit_mix    = breakdown['credit_mix']
    balance_pct   = breakdown['balance_pct']
    debt_to_ai    = breakdown['debt_to_annual_income']
    job_stability = breakdown['job_stability']
    inquiries     = breakdown['inquiries']
    num_cards     = features['Num_Credit_Card']
    num_loans     = features['Num_of_Loan']
    num_banks     = features['Num_Bank_Accounts']

    # INCOME LEVEL INSIGHT (shown before other factors when income is critically low)
    salary_inr = features.get('_salary_inr', 0)
    if salary_inr < 5000:
        concerns.append({
            'title': 'Income Level: Critically Low',
            'message': (
                f'Your monthly income of Rs {salary_inr:,} is below the minimum threshold most lenders use to assess repayment capacity. '
                'No mainstream bank or NBFC will extend credit at this income level. '
                'Building your income is the single most important step before any other financial goal.'
            )
        })
    elif salary_inr < 10000:
        concerns.append({
            'title': 'Income Level: Below Minimum Wage',
            'message': (
                f'Your monthly income of Rs {salary_inr:,} is below the minimum wage threshold in most Indian states. '
                'Most lenders require a minimum of Rs 15,000 to Rs 25,000 per month before they will consider any credit application. '
                'Increasing your income is the most impactful thing you can do for your credit profile right now.'
            )
        })
    elif salary_inr < 15000:
        concerns.append({
            'title': 'Income Level: Below Bank Minimum',
            'message': (
                f'At Rs {salary_inr:,} per month, your income is below the minimum most banks require for credit products like credit cards or personal loans (typically Rs 15,000 to Rs 25,000). '
                'You may qualify for only limited credit products such as secured cards or microfinance loans.'
            )
        })

    # FACTOR 1: Payment History
    has_any_credit = has_cards or features.get('_has_loans', False)

    if num_delays == 0:
        if has_any_credit:
            strengths.append({
                'title': 'Payment History: Perfect Record',
                'message': (
                    'You have a spotless payment record with no missed deadlines. '
                    'This is the single most important factor in your credit score, making up roughly 35% of the total. '
                    'Keep this going no matter what - it is genuinely your strongest financial asset.'
                )
            })
        else:
            strengths.append({
                'title': 'Payment History: No Delays Recorded',
                'message': (
                    'You have reported no missed payments. '
                    'Since you currently have no active credit products, this reflects no negative history, '
                    'which is a neutral-to-positive starting point as you build your credit profile.'
                )
            })
    elif num_delays == 1:
        delay_str = f', averaging {int(delay_days)} days late' if delay_days > 0 else ''
        delay_note = '' if has_any_credit else ' Note: you have no active credit accounts listed — please verify this figure in Sections 3 and 4.'
        concerns.append({
            'title': 'Payment History: One Missed Payment',
            'message': (
                f'You have missed one payment deadline{delay_str}. '
                f'Even one late payment is enough to block you from reaching the Good or Excellent tier, and it can stay on your record for up to 7 years.{delay_note}'
            )
        })
        actions.append({
            'title': 'Set Up Auto-Pay Today',
            'message': (
                'Turn on auto-pay for the full statement balance on every active credit account right away as a safety net. '
                'Put a reminder 5 days before the due date as a backup. One more late mark can lock your score in a lower range for years.'
            )
        })
    elif num_delays == 2:
        delay_str = f', averaging {int(delay_days)} days past due' if delay_days > 0 else ''
        concerns.append({
            'title': 'Payment History: Two Missed Payments',
            'message': (
                f'You have missed {int(num_delays)} payment deadlines{delay_str}. '
                'Two missed payments is no longer seen as a one-off mistake by lenders - it looks like a pattern. This alone caps your score at the Doubtful range no matter how good everything else looks.'
            )
        })
        actions.append({
            'title': 'Not a Single More Late Payment',
            'message': (
                'Set auto-pay for the minimum due on every loan and card account immediately so nothing slips through accidentally. '
                'Then pay the full amount manually before each due date. '
                'From this point forward, zero late payments is the only way out of this range - one more miss and your score gets pushed into Poor territory.'
            )
        })
    else:
        delay_str = f', averaging {int(delay_days)} days past due' if delay_days > 0 else ''
        concerns.append({
            'title': 'Payment History: Chronic Missed Payments',
            'message': (
                f'You have missed {int(num_delays)} payment deadlines{delay_str}. '
                'With this many missed payments, most mainstream lenders will decline your application outright without even looking at the rest of your profile. This needs serious attention.'
            )
        })
        actions.append({
            'title': 'You Need a Structured Recovery Plan',
            'message': (
                'Reach out to a SEBI-registered credit counsellor and get auto-pay enabled on every account today. '
                'Real recovery here takes time - expect 18 to 24 months of consistent on-time payments before you see meaningful improvement. '
                'Start the clock now rather than later.'
            )
        })

    # FACTOR 2: Debt-to-Income Ratio
    if dti < 20:
        strengths.append({
            'title': 'Debt-to-Income Ratio: Very Healthy',
            'message': (
                f'Only {dti:.1f}% of your monthly income is committed to loan EMI repayments, well below the safe threshold of 36%. '
                'This gives you real flexibility - you can take on new credit if needed and you have a cushion if your income takes a hit.'
            )
        })
    elif dti < 36:
        strengths.append({
            'title': 'Debt-to-Income Ratio: Acceptable',
            'message': (
                f'{dti:.1f}% of your income goes to EMI repayments, which falls within the acceptable range most lenders use as their upper threshold. '
                'Be careful about adding more loans on top of this - it is manageable now but does not leave much room.'
            )
        })
    elif dti < 43:
        concerns.append({
            'title': 'Debt-to-Income Ratio: Elevated',
            'message': (
                f'{dti:.1f}% of your monthly income is locked into EMI payments, above the 36% safe benchmark. '
                'Several lenders use 43% as their hard cutoff for approval, which means you are in the caution zone right now.'
            )
        })
        actions.append({
            'title': 'Do Not Add More EMIs Right Now',
            'message': (
                'Hold off on any new loans or EMI commitments until your DTI comes down below 36%. '
                'Any extra money that comes in - salary hike, bonus, gift - should go straight toward your highest-interest loan first.'
            )
        })
    elif dti < 55:
        concerns.append({
            'title': 'Debt-to-Income Ratio: High',
            'message': (
                f'{dti:.1f}% of your income is consumed by loan repayments, well above the safe threshold. '
                'At this level, most lenders will say no to new credit straightaway. Your DTI is one of the main reasons your score is sitting where it is.'
            )
        })
        actions.append({
            'title': 'Focus on Paying Down Debt Fast',
            'message': (
                'List all your loans by interest rate and throw every extra rupee you can at the most expensive one first. '
                'Even cutting your total EMI burden by 15 to 20% over the next year will bring you back into the acceptable range and take real pressure off your score.'
            )
        })
    else:
        concerns.append({
            'title': 'Debt-to-Income Ratio: Critical',
            'message': (
                f'More than half your income ({dti:.1f}%) is going toward loan repayments alone. '
                'That is not sustainable, and any responsible lender will see that immediately. You need to address this urgently.'
            )
        })
        actions.append({
            'title': 'Talk to Your Lenders Now',
            'message': (
                'Call your lenders and ask about restructuring options - extending your loan tenure or a short break on principal payments can provide immediate relief. '
                'At the same time, get a SEBI-registered financial advisor involved to help you map out a realistic 24-month plan to get this under control.'
            )
        })

    # FACTOR 3: Credit Utilization
    if has_cards:
        if util < 10:
            strengths.append({
                'title': 'Credit Utilization: Excellent',
                'message': (
                    f'You are using only {util:.1f}% of your total available credit limit, which is the optimal range. '
                    'This tells lenders you are not relying on credit to get through the month, which is exactly what they want to see.'
                )
            })
        elif util < 30:
            strengths.append({
                'title': 'Credit Utilization: Good',
                'message': (
                    f'Your credit card utilization stands at {util:.1f}%, within the recommended range of under 30%. '
                    'Keeping it under 30% consistently is one of the most reliable habits for a healthy score over time.'
                )
            })
        elif util < 50:
            concerns.append({
                'title': 'Credit Utilization: Moderate',
                'message': (
                    f'You are using {util:.1f}% of your available credit. The benchmark is below 30%, '
                    'and anything above this consistently will start pulling your score down in automated assessments.'
                )
            })
            actions.append({
                'title': 'Bring Your Card Balance Down',
                'message': (
                    'Work on paying down your card balance, or ask your bank for a credit limit increase to bring the ratio down. '
                    f'Getting from {util:.0f}% to below 30% can show up in your score within a single billing cycle.'
                )
            })
        elif util < 70:
            concerns.append({
                'title': 'Credit Utilization: High',
                'message': (
                    f'{util:.1f}% of your credit limit is in use, which is a strong stress signal in credit scoring models. '
                    'Most automated lending systems will flag this level and either decline or offer you worse terms.'
                )
            })
            actions.append({
                'title': 'Make Paying Off Your Card a Priority',
                'message': (
                    f'Stop non-essential card spending and put a fixed amount toward reducing the balance every month. '
                    f'Every 10 percentage points you bring it down from {util:.0f}% typically adds 10 to 20 points to your credit score.'
                )
            })
        else:
            concerns.append({
                'title': 'Credit Utilization: Critical',
                'message': (
                    f'{util:.1f}% credit utilization is in the danger zone. '
                    'At this level, credit models see you as someone who is financially stretched thin, and it puts a hard ceiling on your score.'
                )
            })
            actions.append({
                'title': 'This Is Your Most Urgent Priority Right Now',
                'message': (
                    'Switch to cash or UPI for everything immediately and consider using low-yield savings to pay down the card balance. '
                    'Getting your utilization below 70% needs to happen before anything else - it will have the single biggest impact on your score.'
                )
            })
    else:
        if credit_mix == 0:
            concerns.append({
                'title': 'Credit Utilization: No Credit Card',
                'message': (
                    'Without a credit card, lenders have no way to see how you handle revolving credit on a month-to-month basis. '
                    'That ongoing behavioural signal is one of the things lenders look at most closely.'
                )
            })

    # FACTOR 4: Credit History Length
    # Consistency check: long credit history with no credit products ever is contradictory
    if history_yrs > 1 and not has_cards and credit_mix == 0 and not features.get('_has_loans', False):
        concerns.append({
            'title': 'Credit History: Inconsistent Data',
            'message': (
                f'You have entered {history_yrs:.1f} years of credit history but reported no active credit cards or loans. '
                'Credit history can only be built through active credit products. '
                'If you have had credit products in the past that are now closed, this is valid. '
                'Otherwise, please review your entries in Sections 3 and 4.'
            )
        })

    if history_yrs >= 7:
        strengths.append({
            'title': 'Credit History Length: Established',
            'message': (
                f'You have {history_yrs:.1f} years of active credit history, well above the preferred threshold. '
                'A long track record gives lenders a lot of data to judge you on, and in your case that works strongly in your favour.'
            )
        })
    elif history_yrs >= 4:
        strengths.append({
            'title': 'Credit History Length: Solid',
            'message': (
                f'With {history_yrs:.1f} years of credit history, you are past the point where lenders feel comfortable making a confident decision about you.'
            )
        })
    elif history_yrs >= 2:
        concerns.append({
            'title': 'Credit History Length: Limited',
            'message': (
                f'Your credit history spans only {history_yrs:.1f} years. Most lenders want to see at least 5 years of history before they feel confident assessing your long-term repayment behaviour.'
            )
        })
        actions.append({
            'title': 'Do Not Close Your Oldest Accounts',
            'message': (
                'Even if you barely use your oldest card or loan account, keep it open. '
                'Closing it shortens your credit history, which directly hurts your score. '
                'Time is the only real fix for a short history - you just have to let the clock run.'
            )
        })
    else:
        concerns.append({
            'title': 'Credit History Length: Very Short',
            'message': (
                f'Only {history_yrs:.1f} years of credit history means insufficient behavioral data for lenders to make confident decisions. '
                'When lenders cannot see enough history, they treat it similarly to a bad history - the unknown is a risk they are not willing to take.'
            )
        })
        actions.append({
            'title': 'Start Small and Build From Here',
            'message': (
                'Apply for one secured credit card, use it for a single fixed expense like a utility bill, and set auto-pay to clear the full balance every month. '
                'There is no shortcut to building credit history - it comes from steady, responsible use over time.'
            )
        })

    # FACTOR 5: Credit Mix
    if credit_mix == 2:
        strengths.append({
            'title': 'Credit Mix: Diverse Portfolio',
            'message': (
                'You manage both revolving credit (credit cards) and instalment credit (loans). '
                'Managing both well shows lenders you can handle different kinds of financial obligations, which is a genuine positive in their assessment.'
            )
        })
    elif credit_mix == 1:
        if has_cards and not features['_has_loans']:
            concerns.append({
                'title': 'Credit Mix: Only Revolving Credit',
                'message': (
                    'Your profile includes only credit cards with no active instalment loans. '
                    'Having only one type of credit limits this signal, which accounts for roughly 10% of your total score.'
                )
            })
        elif features['_has_loans'] and not has_cards:
            concerns.append({
                'title': 'Credit Mix: Only Instalment Loans',
                'message': (
                    'Your profile includes only instalment loans with no revolving credit. '
                    'Lenders like to see you handle both types because a credit card gives them ongoing monthly data about your behaviour that a loan alone cannot provide.'
                )
            })
    else:
        concerns.append({
            'title': 'Credit Mix: No Active Credit Products',
            'message': (
                'Right now you have no active credit cards or loans at all. '
                'Without any credit activity, lenders simply have nothing to go on when assessing whether you are a reliable borrower.'
            )
        })
        actions.append({
            'title': 'Get One Credit Product and Use It Well',
            'message': (
                'A secured credit card or a credit-builder loan is a good starting point - both are designed for people with little or no credit history. '
                'Use it for one regular expense and pay the full balance every month. That alone will start building your profile.'
            )
        })

    # FACTOR 6: New Credit Inquiries
    if inquiries == 0:
        strengths.append({
            'title': 'New Credit Inquiries: None',
            'message': (
                'You have made no new credit applications in the assessment period. '
                'No recent applications signals to lenders that you are not scrambling for credit, which is a small but positive mark in your favour.'
            )
        })
    elif inquiries <= 2:
        strengths.append({
            'title': 'New Credit Inquiries: Low',
            'message': (
                f'{inquiries} recent credit application(s) is well within the normal range and will barely register on your score.'
            )
        })
    elif inquiries <= 4:
        concerns.append({
            'title': 'New Credit Inquiries: Moderate',
            'message': (
                f'{inquiries} recent credit inquiries is above the preferred level, which starts to look like you are actively hunting for credit, something lenders associate with financial stress.'
            )
        })
        actions.append({
            'title': 'Take a Break From Applying for Credit',
            'message': (
                'Let your inquiry count settle for at least 3 months before applying for anything new. '
                'When you are ready to apply again, do your homework first and go for the one product you are most likely to get approved for.'
            )
        })
    else:
        concerns.append({
            'title': 'New Credit Inquiries: High',
            'message': (
                f'{inquiries} recent credit inquiries is a red flag. Multiple applications in a short period looks like desperation to lenders, and every single inquiry stays on your record for 2 years.'
            )
        })
        actions.append({
            'title': 'Stop All Applications for at Least 6 Months',
            'message': (
                'Do not apply for any new credit for the next 6 months - the cluster of inquiries needs time to age and lose its impact. '
                'When you start again, use pre-approval tools that do a soft check first so you are not racking up more hard inquiries unnecessarily.'
            )
        })

    # FACTOR 7: Payment in Full vs Minimum
    if has_cards:
        if pays_full:
            strengths.append({
                'title': 'Payment Behavior: Pays Full Balance',
                'message': (
                    'You pay your credit card statement balance in full each month, avoiding revolving interest entirely. '
                    'This is what separates strong credit profiles from average ones. It tells bureaus you can fully meet your obligations every single month.'
                )
            })
        else:
            concerns.append({
                'title': 'Payment Behavior: Minimum or Partial Payments',
                'message': (
                    'You are only paying the minimum or a partial amount, which means the rest rolls over at 36 to 42% annual interest. '
                    'To lenders, this suggests you are not fully in control of your repayments.'
                )
            })
            actions.append({
                'title': 'Work Toward Paying the Full Amount Each Month',
                'message': (
                    'Start by paying at least double the minimum every month - that will begin making a real dent in the principal. '
                    'Once the balance is cleared, make full payment a permanent habit and eliminate that high interest cost completely.'
                )
            })

    # FACTOR 8: Job and Income Stability
    special_occ = breakdown.get('special_occupation', None)

    if special_occ == 'student':
        student_parttime = breakdown.get('student_parttime', False)
        if student_parttime:
            concerns.append({
                'title': 'Income Stability: Student with Part-Time Income',
                'message': (
                    'You are a student with part-time earnings, which is a positive signal compared to a fully dependent student. '
                    'However, part-time income is irregular and not guaranteed, so lenders still treat this as a higher-risk profile. '
                    'Your score is capped in the Fair range until you move into full-time stable employment.'
                )
            })
            actions.append({
                'title': 'Use Your Earnings to Start Building Credit Now',
                'message': (
                    'Since you have some income, apply for a student or entry-level credit card, use it for one small fixed expense, '
                    'and pay the full balance every month. '
                    'This builds your credit history now so your profile is strong by the time you enter full-time employment.'
                )
            })
        else:
            concerns.append({
                'title': 'Income Stability: Student',
                'message': (
                    'As a student with no employment income, lenders have no basis to assess your repayment capacity. '
                    'This makes you a high-risk borrower in any credit assessment, and your score is capped in the Fair range. '
                    'This will change significantly once you enter stable employment.'
                )
            })
            actions.append({
                'title': 'Build Your Profile Before Applying for Credit',
                'message': (
                    'Focus on building a credit history now through a student credit card or a secured card with a low limit. '
                    'Pay it off fully every month and keep the balance low. '
                    'Once you start earning a stable income, your score ceiling will rise significantly.'
                )
            })
    elif special_occ == 'unemployed':
        concerns.append({
            'title': 'Income Stability: Currently Unemployed',
            'message': (
                'Without active employment, lenders have no assurance that your income will continue. '
                'This is one of the most significant risk factors in any credit assessment and caps your score in the Doubtful range.'
            )
        })
        actions.append({
            'title': 'Hold Off on New Credit Until Employed',
            'message': (
                'Avoid applying for any new credit while unemployed - rejections during this period can further hurt your score. '
                'Once you secure stable employment, maintain it for at least 6 months before applying, and let your overall profile rebuild naturally.'
            )
        })
    elif special_occ == 'pensioner':
        strengths.append({
            'title': 'Income Stability: Pensioner',
            'message': (
                'Pension income is fixed and predictable, which is actually a positive signal for lenders. '
                'Unlike employment income, a pension does not stop - it provides guaranteed cash flow for repayments, which supports your creditworthiness.'
            )
        })
    else:
        stability_yrs_map = {
            'More than 10 years': 'more than 10 years',
            '5-10 years': '5 to 10 years',
            '3-5 years': '3 to 5 years',
            '1-3 years': '1 to 3 years',
            'Less than 1 year': 'less than 1 year'
        }
        stability_label = stability_yrs_map.get(job_stability, job_stability)

        if job_stability in ('More than 10 years', '5-10 years'):
            strengths.append({
                'title': 'Income and Job Stability: Strong',
                'message': (
                    f'You have been in your current occupation for {stability_label}. '
                    'A long stint in one occupation tells lenders your income is steady and reliable, which directly supports their confidence in your ability to repay.'
                )
            })
        elif job_stability == '3-5 years':
            strengths.append({
                'title': 'Income and Job Stability: Adequate',
                'message': (
                    f'With {stability_label} in your current occupation, you have shown a reasonable level of stability in your work, which is a modest positive in how lenders assess your profile.'
                )
            })
        elif job_stability == '1-3 years':
            concerns.append({
                'title': 'Income and Job Stability: Moderate Risk',
                'message': (
                    f'You have been in your current occupation for {stability_label}. Shorter tenures are treated as a moderate risk signal, especially when you are applying for larger loans like a home loan.'
                )
            })
            actions.append({
                'title': 'Wait Before Applying for Big Loans',
                'message': (
                    'Try to clock at least 2 years in your current role before going for a home loan or large personal loan. '
                    'Most mortgage lenders will want to see that minimum before they even consider your application.'
                )
            })
        else:
            concerns.append({
                'title': 'Income and Job Stability: High Risk',
                'message': (
                    f'You have been in your current occupation for {stability_label}. Being less than a year into your current role raises genuine questions about how stable your income is and whether you can keep up with repayments long-term.'
                )
            })
            actions.append({
                'title': 'Stay Put and Build Your Track Record',
                'message': (
                    'Try to stay in your current role for at least 12 to 24 months before applying for any significant credit. '
                    'If you are self-employed, keep your ITR filings current and organised - lenders typically want 2 to 3 years of documented income proof.'
                )
            })

    # FACTOR 9: Savings Rate
    if savings_rate >= 20:
        strengths.append({
            'title': 'Savings Discipline: Excellent',
            'message': (
                f'You save or invest {savings_rate:.1f}% of your monthly income, well above the recommended 15% benchmark. '
                'A high savings rate means you have a real cushion to fall back on if something unexpected comes up, which keeps your payments safe.'
            )
        })
    elif savings_rate >= 12:
        strengths.append({
            'title': 'Savings Discipline: Good',
            'message': (
                f'Saving {savings_rate:.1f}% of your monthly income meets the recommended minimum benchmark, which shows that your spending is genuinely within your means.'
            )
        })
    elif savings_rate >= 5:
        concerns.append({
            'title': 'Savings Discipline: Below Target',
            'message': (
                f'Your savings rate of {savings_rate:.1f}% is below the recommended 15%. '
                'With such a thin cushion, even a single unexpected expense could push you toward expensive credit, which then makes everything harder to manage.'
            )
        })
        actions.append({
            'title': 'Automate Your Savings Before You Can Spend It',
            'message': (
                'Set up an automatic transfer the day your salary hits - even 10% to start is meaningful. '
                'Nudge it up by 2% every few months. Your first real target should be 3 months of living expenses sitting in a separate account.'
            )
        })
    else:
        concerns.append({
            'title': 'Savings Discipline: Critically Low',
            'message': (
                f'A savings rate of {savings_rate:.1f}% provides almost no financial cushion. '
                'Without any savings buffer, even a minor surprise expense can force you to borrow, which adds to your debt and puts your payment record at risk.'
            )
        })
        actions.append({
            'title': 'Build a Safety Net First',
            'message': (
                'Before anything else, focus on getting one month of total expenses into a separate savings account. '
                'Even Rs 1,000 to Rs 2,000 a month through a recurring deposit will get you there faster than you think.'
            )
        })

    # FACTOR 10: Bank Balance Adequacy
    if balance_pct >= 80:
        strengths.append({
            'title': 'Liquidity: Strong Bank Balance',
            'message': (
                f'Your average bank balance is approximately {balance_pct:.0f}% of your monthly income. '
                'Having that buffer means you can handle your EMI and card payments without stress, even if your salary comes in a few days late.'
            )
        })
    elif balance_pct >= 30:
        strengths.append({
            'title': 'Liquidity: Adequate Bank Balance',
            'message': (
                f'Your bank balance represents about {balance_pct:.0f}% of your monthly income, which gives you a reasonable buffer to handle your regular payment commitments.'
            )
        })
    elif balance_pct >= 10:
        concerns.append({
            'title': 'Liquidity: Low Bank Balance',
            'message': (
                f'Your average bank balance is only about {balance_pct:.0f}% of your monthly income. '
                'If your salary is delayed even slightly or an unexpected expense comes up, you could easily miss a scheduled payment.'
            )
        })
        actions.append({
            'title': 'Keep a Dedicated Payment Buffer in Your Account',
            'message': (
                'Set a personal rule to always keep at least one full month of combined EMI and card payments sitting in your account. '
                'Treat it as untouchable - that buffer is what keeps your payment record clean when things get unpredictable.'
            )
        })
    else:
        concerns.append({
            'title': 'Liquidity: Critically Low Bank Balance',
            'message': (
                f'Your bank balance of approximately {balance_pct:.0f}% of monthly income is critically low. '
                'If your salary is even a day or two late hitting your account, the chances of missing a payment become very real - and a missed payment is the biggest single penalty in credit scoring.'
            )
        })
        actions.append({
            'title': 'Build a Cash Reserve Before Anything Else',
            'message': (
                'This month, before any non-essential spending, move a fixed amount into a separate account specifically as a payment buffer. '
                'Your only goal right now is to get that account up to one full month of EMI obligations.'
            )
        })

    # FACTOR 11: Outstanding Debt Relative to Annual Income
    if debt_to_ai < 20:
        strengths.append({
            'title': 'Total Debt Load: Low',
            'message': (
                f'Your total outstanding loan balance is {debt_to_ai:.0f}% of your annual income — '
                'well within healthy limits. Even if your income dips temporarily, '
                'your overall borrowing is small enough that lenders will not see you as over-leveraged.'
            )
        })
    elif debt_to_ai < 60:
        # Only show if DTI is not already flagged — avoids piling on when both are moderate
        if dti < 36:
            concerns.append({
                'title': 'Total Debt Load: Moderate',
                'message': (
                    f'Your total outstanding balance is {debt_to_ai:.0f}% of your annual income. '
                    'Your monthly repayments are manageable right now, but the overall size of your debt '
                    'limits room for additional borrowing. Gradual prepayments will strengthen this over time.'
                )
            })
    elif debt_to_ai < 80:
        concerns.append({
            'title': 'Total Debt Load: High',
            'message': (
                f'Your total outstanding balance is {debt_to_ai:.0f}% of your annual income. '
                'This is a separate concern from your monthly repayment affordability — it tells lenders '
                'that even if you are keeping up with EMIs today, you are carrying a heavy overall debt burden. '
                'Most lenders will hesitate to extend further credit until this comes down significantly.'
            )
        })
        actions.append({
            'title': 'Direct Any Surplus Toward Principal Prepayment',
            'message': (
                'Any bonus, increment, or windfall should go toward prepaying your highest-interest loan first. '
                f'Getting your outstanding balance from {debt_to_ai:.0f}% to below 60% of your annual income '
                'will reduce the leverage risk lenders see in your profile — separate from improving your monthly DTI.'
            )
        })
    else:
        concerns.append({
            'title': 'Total Debt Load: Severely Over-Leveraged',
            'message': (
                f'Your total outstanding debt is {debt_to_ai:.0f}% of your annual income. '
                'This signals over-leverage independently of whether your monthly EMIs are affordable right now. '
                'Lenders assess both: you can be current on every payment and still be declined '
                'because the sheer size of your debt relative to income makes you a high risk for any new credit.'
            )
        })
        actions.append({
            'title': 'Focus on Reducing Principal, Not Just Paying EMIs',
            'message': (
                'Staying current on EMIs is necessary but not enough here. '
                'You need to actively reduce the outstanding principal through prepayments or part-closures. '
                'Consider speaking with a SEBI-registered financial advisor to build a structured paydown plan.'
            )
        })
 

    # FACTOR 12: Number of Bank Accounts
    if num_banks == 0:
        concerns.append({
            'title': 'Banking Relationship: No Bank Accounts',
            'message': (
                'You currently have no active bank accounts. Without any banking activity, lenders have no way to verify your cash flow, which makes processing any credit application extremely difficult.'
            )
        })
        actions.append({
            'title': 'Open a Bank Account as Soon as Possible',
            'message': (
                'Open a savings account with a proper scheduled bank and start routing all your income through it. '
                'Six months of regular salary credits is usually the minimum lenders need before they will even look at a basic credit card application.'
            )
        })
    elif num_banks > 5:
        concerns.append({
            'title': 'Banking Relationship: Too Many Accounts',
            'message': (
                f'Maintaining {int(num_banks)} bank accounts spreads your financial activity too thin and increases the risk of dormant accounts quietly racking up maintenance charges.'
            )
        })
        actions.append({
            'title': 'Simplify Down to 2 or 3 Accounts',
            'message': (
                'Pick 2 to 3 accounts you actually use regularly for salary, spending, and savings, and close or convert the rest. '
                'Keeping your banking focused makes your financial picture much clearer and easier for lenders to read.'
            )
        })
    else:
        if num_banks <= 3:
            strengths.append({
                'title': 'Banking Relationship: Well-Maintained',
                'message': (
                    f'You maintain {int(num_banks)} active bank account(s), giving lenders a clear and trackable picture of your finances. '
                    'Keeping your banking focused with consistent activity is viewed positively when your credit is being assessed.'
                )
            })

    # FACTOR 13: Number of Credit Cards
    if has_cards:
        if num_cards > 6:
            concerns.append({
                'title': 'Number of Credit Cards: Excessive',
                'message': (
                    f'Holding {int(num_cards)} credit cards makes it genuinely harder to track all your payments and bills, and can signal to lenders that you are leaning too heavily on borrowed money.'
                )
            })
            actions.append({
                'title': 'Trim Down to 3 or 4 Cards',
                'message': (
                    'Start by closing cards with the highest fees and the lowest limits, but always keep your oldest card to protect your history length. '
                    '3 to 4 well-managed cards with low balances will always look better than having many cards spread thin.'
                )
            })
        elif num_cards <= 3:
            strengths.append({
                'title': 'Number of Credit Cards: Manageable',
                'message': (
                    f'{int(num_cards)} credit card(s) is a perfectly manageable number that makes it easy to stay on top of your spending and never miss a payment deadline.'
                )
            })

    # Fallbacks
    if not strengths:
        strengths.append({
            'title': 'You Have Already Taken the First Step',
            'message': (
                'Your profile has real room for improvement, but the fact that you are assessing it now is genuinely the most important first step. '
                'Every change you make from here will show up in your score over the next 12 to 24 months.'
            )
        })

    if not concerns:
        concerns.append({
            'title': 'Your Profile Is Strong - Keep It That Way',
            'message': (
                'You are in a good position right now. One missed payment or a sudden jump in card spending can undo years of good behaviour surprisingly quickly. '
                'Check your official credit report at least once a year through CIBIL or Experian to catch any errors or anything unusual before it becomes a problem.'
            )
        })

    if not actions:
        actions.append({
            'title': 'Keep Doing What You Are Doing',
            'message': (
                'Your habits are working - do not change them. Check your credit report once a year and try not to apply for multiple new credit products within the same year.'
            )
        })

    return {'strengths': strengths, 'concerns': concerns, 'actions': actions}


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
        "prompt": f"""You are a strict Indian financial advisor. You ONLY answer questions about credit scores, budgeting, loans, debt, savings, credit cards, and financial planning in the Indian context.

{context_instruction}

STRICT RULES:
1. Answer ONLY the question asked. Do NOT add any unrequested information.
2. Do NOT assume or invent anything about the user's financial situation. You have NO information about their credit score, income, loans, or history unless they tell you in this conversation.
3. If the question is about another country (e.g. US, UK), redirect: "I focus on Indian finance. In India, [answer]."
4. Never start with phrases like "Sure", "Great question", "Of course", or "Here's the answer". Start directly with the answer.
5. Use Indian context — rupees, CIBIL, RBI, Indian banks.
6. Be concise. Keep answers under 150 words unless detail is asked for.
7. Always complete every sentence. Never stop mid-answer.
8. Use the information if it is provided in the question.

Answer:""",
        "stream": False,
        "options": {
            "temperature": 0.6,
            "num_predict": 400,
            "top_p": 0.9
        }
    }

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=120,
        )

        if response.status_code == 200:
            answer = response.json().get("response", "").strip()

            bad_phrases = ["Question:", "User question:", "Next question:",
                           "You might also", "For example, if", "\nQuestion", "User:", "Assistant:"]
            for phrase in bad_phrases:
                if phrase in answer:
                    answer = answer.split(phrase)[0].strip()

            # Strip common preamble openers the model adds
            preambles = [
                "Sure, here's the answer to your question:",
                "Sure, here is the answer to your question:",
                "Sure! Here's the answer:",
                "Great question!",
                "Of course!",
                "Of course,",
                "Certainly!",
                "Certainly,",
                "Here's the answer:",
                "Here is the answer:",
                "Happy to help!",
            ]
            for p in preambles:
                if answer.startswith(p):
                    answer = answer[len(p):].strip()

            # Remove sentences where model hallucinated about user's personal data
            hallucination_triggers = [
                'your credit score is', 'your score is', 'your financial situation',
                'your credit history', 'you have a history of', 'you have missed',
                'your credit utilization is', 'your debt', 'your income',
                'based on your profile', 'looking at your', 'according to your',
                'your current score', 'your cibil', 'you seem to have',
                'it appears you', 'it looks like you',
            ]
            cleaned_sentences = []
            for sentence in answer.replace('!', '.').replace('?', '.').split('.'):
                s = sentence.strip().lower()
                if s and not any(trigger in s for trigger in hallucination_triggers):
                    cleaned_sentences.append(sentence.strip())
            if cleaned_sentences:
                answer = '. '.join(cleaned_sentences).strip()
                if answer and answer[-1] not in '.!?':
                    answer += '.'

            # Ensure answer ends on a complete sentence — no mid-sentence cutoff
            if answer and answer[-1] not in '.!?':
                last_stop = max(answer.rfind('.'), answer.rfind('!'), answer.rfind('?'))
                if last_stop > len(answer) // 2:
                    answer = answer[:last_stop + 1].strip()

            if any(word in input_lower for word in ['1 line', 'one line', 'brief', 'shortly', 'quick']):
                sentences = answer.split('.')
                answer = sentences[0] + '.' if sentences else answer

            if len(answer) < 15:
                return "Could you rephrase that? I'm here to help with credit scores, budgeting, and financial planning."

            return answer
        else:
            return "Something went wrong. Try again?"

    except requests.exceptions.Timeout:
        return "Taking too long. Try a simpler question or make sure Ollama is running."
    except requests.exceptions.ConnectionError:
        return "Can't connect to the chatbot. Make sure Ollama is running with: ollama serve"
    except Exception as e:
        return f"Error: {str(e)}"

# NAVIGATION HELPERS
def go_to_home():
    st.session_state.page = 'home'
    st.session_state.section = 1
    st.session_state.show_chatbot = False

def go_to_assessment():
    st.session_state.page = 'assessment'
    st.session_state.section = 1

def next_section(): st.session_state.section += 1
def prev_section(): st.session_state.section -= 1

def reset_assessment():
    st.session_state.answers = {}
    st.session_state.section = 1

def toggle_chatbot():
    st.session_state.show_chatbot = not st.session_state.show_chatbot
    if st.session_state.show_chatbot and not st.session_state.messages:
        st.session_state.messages = [{
            'role': 'assistant',
            'content': (
                'Hi! I can help with credit scores, budgeting, debt management, '
                'and financial planning. What would you like to know?'
            )
        }]

def clear_chat():
    st.session_state.messages = [{
        'role': 'assistant',
        'content': (
            'Hi! I can help with credit scores, budgeting, debt management, '
            'and financial planning. What would you like to know?'
        )
    }]


# PAGE: HOME
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


# ASSESSMENT WRAPPER
def show_assessment():
    sections = ['Personal', 'Income', 'Credit', 'Loans', 'Payment History']
    progress_html = '<div class="progress-bar">'
    for i, sec in enumerate(sections, 1):
        if   i < st.session_state.section: cls = 'progress-step completed'
        elif i == st.session_state.section: cls = 'progress-step active'
        else:                               cls = 'progress-step'
        progress_html += f'<div class="{cls}">{sec}</div>'
    progress_html += '</div>'
    st.markdown(progress_html, unsafe_allow_html=True)

    s = st.session_state.section
    if   s == 1: show_section_1()
    elif s == 2: show_section_2()
    elif s == 3: show_section_3()
    elif s == 4: show_section_4()
    elif s == 5: show_section_5()
    elif s == 6:
        if st.session_state.show_chatbot: show_chatbot()
        else:                             show_results()


# SECTION 1: Personal Information
def show_section_1():
    st.markdown("""
    <div class="section-card">
        <h2 class="section-title">Personal Information</h2>
        <p class="section-description">Basic information about you</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=80,
                               value=st.session_state.answers.get('age', 30))
        st.session_state.answers['age'] = age

    with col2:
        occupation = st.selectbox(
            "Occupation",
            options=list(OCCUPATION_MAP.keys()),
            index=list(OCCUPATION_MAP.keys()).index(
                st.session_state.answers.get('occupation', 'Other'))
        )
        st.session_state.answers['occupation'] = occupation

    # Special occupations that hide job stability question
    SPECIAL_OCCUPATIONS = {'student', 'unemployed', 'pensioner'}

    if occupation == 'Other':
        st.caption("If your occupation is not listed above, enter it here (e.g., Student, Pensioner, Unemployed, Freelancer, Farmer, or any other role).")
        other_occ = st.text_input(
            "Please specify your occupation (Required)",
            value=st.session_state.answers.get('other_occupation_text', ''),
            help="If your occupation is not listed in the dropdown, please enter it manually (e.g., Student, Pensioner, Unemployed, or any other role)."
        )
        if other_occ:
            if not other_occ.replace(' ', '').isalpha():
                st.error("Please enter only alphabetical characters (no numbers or symbols).")
                st.session_state.answers['other_occupation_text'] = ''
            elif other_occ.strip().lower() in [o.lower() for o in OCCUPATION_MAP.keys() if o != 'Other']:
                st.error(
                    f'"{other_occ}" is already available in the dropdown above. '
                    'Please select it from the Occupation dropdown instead.'
                )
                st.session_state.answers['other_occupation_text'] = ''
                st.session_state.answers['special_occupation'] = None
            elif other_occ.strip().lower() not in VALID_OTHER_OCCUPATIONS:
                st.error(
                    f'"{other_occ}" is not a recognised occupation. '
                    'Please enter a valid job title (e.g. Freelancer, Farmer, Nurse, Retired, Student).'
                )
                st.session_state.answers['other_occupation_text'] = ''
                st.session_state.answers['special_occupation'] = None
            else:
                st.session_state.answers['other_occupation_text'] = other_occ
                # Detect special occupation (case-insensitive)
                occ_lower = other_occ.strip().lower()
                if occ_lower in SPECIAL_OCCUPATIONS:
                    st.session_state.answers['special_occupation'] = occ_lower
                else:
                    st.session_state.answers['special_occupation'] = None
        else:
            st.session_state.answers['other_occupation_text'] = ''
            st.session_state.answers['special_occupation'] = None
    else:
        st.session_state.answers['special_occupation'] = None

    st.markdown("<div class='spacer-small'></div>", unsafe_allow_html=True)

    # Determine if job stability question should be shown
    special_occ = st.session_state.answers.get('special_occupation', None)
    show_stability = (occupation != 'Other') or (special_occ not in SPECIAL_OCCUPATIONS)

    if show_stability:
        income_stability = st.selectbox(
            "How long have you been in your current occupation?",
            options=["Less than 1 year","1-3 years","3-5 years","5-10 years","More than 10 years"],
            index=["Less than 1 year","1-3 years","3-5 years","5-10 years","More than 10 years"].index(
                st.session_state.answers.get('income_stability', '3-5 years'))
        )
        st.session_state.answers['income_stability'] = income_stability
    else:
        # Keep a neutral default in session state so calculate_features doesn't error
        st.session_state.answers['income_stability'] = '3-5 years'
        # Show a note so user understands why the question is hidden
        if special_occ == 'student':
            st.info("Job stability is not applicable for students.")
            # Ask if student does part-time work
            parttime = st.radio(
                "Do you earn from any part-time work?",
                options=["Yes", "No"],
                index=0 if st.session_state.answers.get('student_parttime', False) else 1,
                horizontal=True
            )
            st.session_state.answers['student_parttime'] = (parttime == "Yes")
        elif special_occ == 'unemployed':
            st.info("Job stability is not applicable for unemployed individuals.")
            st.session_state.answers['student_parttime'] = False
        elif special_occ == 'pensioner':
            st.info("Job stability is not applicable for pensioners.")
            st.session_state.answers['student_parttime'] = False

    st.markdown("<div class='spacer-medium'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col3:
        if st.button("Next", type="primary", use_container_width=True):
            other_text = st.session_state.answers.get('other_occupation_text', '').strip()
            if occupation == 'Other' and not other_text:
                st.error('Please specify your occupation.')
            elif occupation == 'Other' and other_text.lower() in [o.lower() for o in OCCUPATION_MAP.keys() if o != 'Other']:
                st.error(
                    f'"{other_text}" is already available in the dropdown above. '
                    'Please select it from the Occupation dropdown instead.'
                )
            elif occupation == 'Other' and other_text.lower() not in VALID_OTHER_OCCUPATIONS:
                st.error(
                    f'"{other_text}" is not a recognised occupation. '
                    'Please enter a valid job title (e.g. Freelancer, Farmer, Nurse, Retired, Student).'
                )
            else:
                next_section()
                st.rerun()


# SECTION 2: Income and Financial Health
def show_section_2():
    st.markdown("""
    <div class="section-card">
        <h2 class="section-title">Income and Financial Health</h2>
        <p class="section-description">Your income and savings information (all amounts in Rs)</p>
    </div>
    """, unsafe_allow_html=True)

    # Determine salary label and help text based on special occupation
    special_occ = st.session_state.answers.get('special_occupation', None)
    student_parttime = st.session_state.answers.get('student_parttime', False)

    if special_occ == 'unemployed':
        salary_label = "Monthly Income from Any Source (Rs)"
        salary_help  = "Total money received each month from all sources — family support, freelance work, rental income, or anything else"
    elif special_occ == 'student' and student_parttime:
        salary_label = "Total Monthly Earnings (Rs)"
        salary_help  = "Combined amount from part-time work, family support, scholarships, or any other source you rely on each month"
    elif special_occ == 'student' and not student_parttime:
        salary_label = "Monthly Financial Support Received (Rs)"
        salary_help  = "Total amount received each month from family, scholarships, stipends, or any other source of financial support"
    elif special_occ == 'pensioner':
        salary_label = "Monthly Pension Amount (Rs)"
        salary_help  = "Your total monthly pension payout after any deductions"
    else:
        salary_label = "Monthly Take-Home Salary (Rs)"
        salary_help  = "Your monthly in-hand salary after all deductions"

    col1, col2 = st.columns(2)
    with col1:
        salary = st.number_input(
            salary_label,
            min_value=0, max_value=10000000,
            value=st.session_state.answers.get('monthly_salary', 0),
            step=1000,
            help=salary_help
        )
        st.session_state.answers['monthly_salary'] = salary

    with col2:
        savings = st.number_input(
            "Monthly Savings and Investments (Rs)",
            min_value=0, max_value=10000000,
            value=st.session_state.answers.get('monthly_savings', 0),
            step=500,
            help="Amount you save or invest each month (SIP, FD, RD, etc.)"
        )
        st.session_state.answers['monthly_savings'] = savings

    st.markdown("<div class='spacer-small'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        bank_accounts = st.number_input(
            "Number of Bank Accounts",
            min_value=0, max_value=15,
            value=st.session_state.answers.get('num_bank_accounts', 2)
        )
        st.session_state.answers['num_bank_accounts'] = bank_accounts

    with col2:
        balance = st.number_input(
            "Average Bank Balance (Rs)",
            min_value=0, max_value=100000000,
            value=st.session_state.answers.get('monthly_balance', 0),
            step=1000,
            help="Typical combined balance across all your bank accounts"
        )
        st.session_state.answers['monthly_balance'] = balance

    st.markdown("<div class='spacer-medium'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("Back", use_container_width=True):
            prev_section(); st.rerun()

    with col3:
        if st.button("Next", type="primary", use_container_width=True):
            errors = []
            salary = st.session_state.answers.get('monthly_salary', 0)
            savings = st.session_state.answers.get('monthly_savings', 0)

            if salary == 0:
                errors.append('Monthly salary cannot be 0.')
            if 0 < salary < 1000:
                errors.append(
                    f'A monthly income of Rs {salary:,} is not a valid figure. '
                    'Please enter your actual monthly income.'
                )
            if salary > 0 and savings > salary:
                errors.append(
                    f'Monthly savings (Rs {savings:,}) cannot exceed monthly salary '
                    f'(Rs {salary:,}). Please re-enter both values.'
                )
            if salary > 0 and savings > salary * 0.80:
                errors.append(
                    f'Saving {savings/salary*100:.0f}% of income is unusually high. '
                    'Please verify this figure.'
                )
            for err in errors:
                st.error(err)
            if not errors:
                next_section(); st.rerun()


# SECTION 3: Credit Cards and History
def show_section_3():
    st.markdown("""
    <div class="section-card">
        <h2 class="section-title">Credit Cards and History</h2>
        <p class="section-description">Your credit card usage and credit history</p>
    </div>
    """, unsafe_allow_html=True)

    age = st.session_state.answers.get('age', 30)
    max_credit_history = float(age - 18)

    credit_history = st.slider(
        "How many years have you been using any credit products?",
        min_value=0.0, max_value=max(max_credit_history, 0.5),
        value=min(
            float(st.session_state.answers.get('credit_history_years', 5.0)),
            max_credit_history
        ),
        step=0.5,
        help=f"Cannot exceed {max_credit_history:.0f} years (your age minus 18)"
    )
    st.session_state.answers['credit_history_years'] = credit_history

    st.markdown("<div class='spacer-small'></div>", unsafe_allow_html=True)

    has_cards = st.radio(
        "Do you have any credit cards?",
        options=["Yes", "No"],
        index=0 if st.session_state.answers.get('has_credit_card', False) else 1,
        horizontal=True
    )
    st.session_state.answers['has_credit_card'] = (has_cards == "Yes")

    if st.session_state.answers['has_credit_card']:
        salary = st.session_state.answers.get('monthly_salary', 0)

        col1, col2 = st.columns(2)
        with col1:
            num_cards = st.number_input(
                "Number of Credit Cards",
                min_value=1, max_value=20,
                value=max(1, st.session_state.answers.get('num_credit_cards', 1))
            )
            st.session_state.answers['num_credit_cards'] = num_cards

        with col2:
            card_spending = st.number_input(
                "Monthly Credit Card Spending (Rs)",
                min_value=0, max_value=10000000,
                value=st.session_state.answers.get('monthly_card_spending', 0),
                step=500
            )
            st.session_state.answers['monthly_card_spending'] = card_spending

        pays_full = st.radio(
            "Do you pay the full bill or minimum due?",
            options=["Full amount every time", "Minimum due or partial"],
            index=0 if st.session_state.answers.get('pays_full', True) else 1
        )
        st.session_state.answers['pays_full'] = (pays_full == "Full amount every time")
    else:
        st.session_state.answers['num_credit_cards']     = 0
        st.session_state.answers['monthly_card_spending'] = 0
        st.session_state.answers['pays_full']             = True

    st.markdown("<div class='spacer-small'></div>", unsafe_allow_html=True)

    has_card_now = st.session_state.answers.get('has_credit_card', False)

    if has_card_now:
        col1, col2 = st.columns(2)
        with col1:
            inquiries = st.number_input(
                "Credit applications in past 6 months",
                min_value=0, max_value=20,
                value=st.session_state.answers.get('num_credit_inquiries', 0),
                help="Number of times you applied for any new credit product"
            )
            st.session_state.answers['num_credit_inquiries'] = inquiries

        with col2:
            limit_changes_range = st.selectbox(
                "Credit limit changes in past 2 years",
                options=["Never","1-2 times","3-4 times","5+ times"],
                index=["Never","1-2 times","3-4 times","5+ times"].index(
                    st.session_state.answers.get('limit_changes_range', 'Never'))
            )
            st.session_state.answers['limit_changes_range'] = limit_changes_range
            limit_map = {"Never": 0, "1-2 times": 1.5, "3-4 times": 3.5, "5+ times": 7}
            st.session_state.answers['num_credit_limit_changes'] = limit_map[limit_changes_range]

    else:
        inquiries = st.number_input(
            "Credit applications in past 6 months",
            min_value=0, max_value=20,
            value=st.session_state.answers.get('num_credit_inquiries', 0),
            help="Number of times you applied for any new credit product (loans, cards, etc.)"
        )
        st.session_state.answers['num_credit_inquiries'] = inquiries
        st.session_state.answers['limit_changes_range'] = 'Never'
        st.session_state.answers['num_credit_limit_changes'] = 0

    st.markdown("<div class='spacer-medium'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("Back", use_container_width=True):
            prev_section(); st.rerun()

    with col3:
        if st.button("Next", type="primary", use_container_width=True):
            errors   = []
            salary   = st.session_state.answers.get('monthly_salary', 0)
            spending = st.session_state.answers.get('monthly_card_spending', 0)
            hist     = st.session_state.answers.get('credit_history_years', 0)

            if st.session_state.answers.get('has_credit_card') and spending > salary:
                errors.append(
                    f'Monthly card spending (Rs {spending:,}) cannot exceed your '
                    f'monthly salary (Rs {salary:,}). Please re-enter.'
                )
            if hist > max_credit_history + 0.1:
                errors.append(
                    f'Credit history ({hist:.1f} years) cannot exceed '
                    f'{max_credit_history:.0f} years (your age minus 18).'
                )
            has_card = st.session_state.answers.get('has_credit_card', False)
            # Credit history requires at least one credit product to have existed
            if hist > 2 and not has_card:
                # They may have loans — we check in section 4, but warn now if no card entered yet
                # Only warn for very high history with no card (loans will be checked in section 4)
                pass  # Handled at results stage — user may have loans (section 4 not seen yet)
            for err in errors:
                st.error(err)
            if not errors:
                next_section(); st.rerun()


# SECTION 4: Loans and Debt
def show_section_4():
    st.markdown("""
    <div class="section-card">
        <h2 class="section-title">Loans and Debt</h2>
        <p class="section-description">Information about your active loans (all amounts in Rs)</p>
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
                min_value=1, max_value=10,
                value=max(1, st.session_state.answers.get('num_loans', 1))
            )
            st.session_state.answers['num_loans'] = num_loans

        with col2:
            loan_type = st.selectbox(
                "Primary Loan Type",
                options=list(LOAN_TYPE_MAP.keys()),
                index=list(LOAN_TYPE_MAP.keys()).index(
                    st.session_state.answers.get('loan_type', 'Not Specified'))
            )
            st.session_state.answers['loan_type'] = loan_type

        if loan_type == 'Not Specified':
            spec_loan = st.text_input(
                "Please specify your loan type (Required)",
                value=st.session_state.answers.get('specified_loan_text', ''),
                help="E.g., Education Loan, Business Loan (alphabets only)"
            )
            if spec_loan:
                spec_clean = spec_loan.strip()
                if not spec_clean.replace(' ', '').isalpha():
                    st.error("Please enter only alphabetical characters")
                    st.session_state.answers['specified_loan_text'] = ''
                else:
                    existing_lower = [lt.lower() for lt in LOAN_TYPE_MAP.keys()
                                      if lt != 'Not Specified']
                    if spec_clean.lower() in existing_lower:
                        st.error(f"'{spec_loan}' is already in the list - please select it from the dropdown.")
                        st.session_state.answers['specified_loan_text'] = ''
                    else:
                        st.session_state.answers['specified_loan_text'] = spec_clean
            else:
                st.session_state.answers['specified_loan_text'] = ''

        salary = st.session_state.answers.get('monthly_salary', 1)

        outstanding = st.number_input(
            "Total Outstanding Loan Amount (Rs)",
            min_value=0, max_value=100000000,
            value=st.session_state.answers.get('outstanding_debt', 0),
            step=10000,
            help="Total principal still owed across all active loans"
        )
        st.session_state.answers['outstanding_debt'] = outstanding

        emi = st.number_input(
            "Total Monthly EMI (Rs)",
            min_value=0, max_value=10000000,
            value=st.session_state.answers.get('total_emi', 0),
            step=500,
            help="Combined EMI amount for all active loans per month"
        )
        st.session_state.answers['total_emi'] = emi

        if salary > 0 and emi > 0:
            dti_preview = (emi / salary) * 100
            if dti_preview > 70:
                st.error(
                    f'Your EMI of Rs {emi:,} is {dti_preview:.0f}% of your monthly salary. '
                    'This is an extremely high debt burden (above 70% DTI). '
                    'Please verify these numbers are correct before proceeding.'
                )
            elif dti_preview > 50:
                st.warning(
                    f'Your EMI represents {dti_preview:.0f}% of your monthly salary - '
                    'this is above the safe threshold of 50% and indicates high financial stress.'
                )

        if outstanding > 0 and emi == 0:
            st.error(
                'You have entered an outstanding loan balance but Rs 0 as your monthly EMI. '
                'An active loan always has a monthly repayment. Please enter your actual EMI amount.'
            )

    else:
        st.session_state.answers['num_loans']         = 0
        st.session_state.answers['loan_type']         = 'Not Specified'
        st.session_state.answers['outstanding_debt']  = 0
        st.session_state.answers['total_emi']         = 0

    st.markdown("<div class='spacer-medium'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("Back", use_container_width=True):
            prev_section(); st.rerun()

    with col3:
        if st.button("Next", type="primary", use_container_width=True):
            errors  = []
            salary  = st.session_state.answers.get('monthly_salary', 1)
            emi     = st.session_state.answers.get('total_emi', 0)
            savings = st.session_state.answers.get('monthly_savings', 0)
            debt    = st.session_state.answers.get('outstanding_debt', 0)

            if st.session_state.answers.get('has_loans'):
                if (st.session_state.answers.get('loan_type') == 'Not Specified' and
                        not st.session_state.answers.get('specified_loan_text', '').strip()):
                    errors.append('Please specify your loan type.')
                if debt == 0:
                    errors.append('Outstanding loan balance cannot be 0 if you have active loans.')
                if emi == 0 and debt > 0:
                    errors.append(
                        'Monthly EMI cannot be Rs 0 if you have an outstanding loan balance. '
                        'Please enter your actual monthly EMI.'
                    )
                # Minimum sensible EMI check — EMI must be at least 0.1% of outstanding debt
                if emi > 0 and debt > 0 and emi < debt * 0.001:
                    errors.append(
                        f'Your EMI of Rs {emi:,} is unrealistically low for an outstanding balance of '
                        f'Rs {debt:,}. Please verify your EMI amount.'
                    )
                if emi > salary:
                    errors.append(
                        f'Total monthly EMI (Rs {emi:,}) cannot exceed your monthly salary '
                        f'(Rs {salary:,}). Please verify your inputs.'
                    )
                total_outflow = emi + savings
                if total_outflow > salary:
                    errors.append(
                        f'Your combined EMI (Rs {emi:,}) and savings (Rs {savings:,}) = '
                        f'Rs {total_outflow:,}, which exceeds your monthly salary of '
                        f'Rs {salary:,}. Please correct these figures.'
                    )

            for err in errors:
                st.error(err)
            if not errors:
                next_section(); st.rerun()


# SECTION 5: Payment History
def show_section_5():
    st.markdown("""
    <div class="section-card">
        <h2 class="section-title">Payment History</h2>
        <p class="section-description">This is the most important factor in your credit score</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        num_delays = st.number_input(
            "Missed payment deadlines in past year",
            min_value=0, max_value=50,
            value=st.session_state.answers.get('num_delayed_payments', 0),
            help="Total number of times you missed any loan EMI or credit card payment"
        )
        st.session_state.answers['num_delayed_payments'] = num_delays

    with col2:
        if num_delays > 0:
            delay_days = st.number_input(
                "Average number of days late",
                min_value=1, max_value=90,
                value=max(1, st.session_state.answers.get('delay_days', 7))
            )
            st.session_state.answers['delay_days'] = delay_days
        else:
            st.session_state.answers['delay_days'] = 0

    has_cards = st.session_state.answers.get('has_credit_card', False)
    has_loans = st.session_state.answers.get('has_loans', False)
    if num_delays > 0 and not has_cards and not has_loans:
        st.warning(
            "You've reported missed payments but haven't listed any active credit cards or loans. "
            "If the delay is valid, proceed; otherwise, review Sections 3 and 4 before submitting."
        )

    st.markdown("<div class='spacer-medium'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Back", use_container_width=True):
            prev_section(); st.rerun()
    with col2:
        if st.button("Reset All", use_container_width=True):
            reset_assessment(); st.rerun()
    with col3:
        if st.button("Calculate My Score", type="primary", use_container_width=True):
            next_section(); st.rerun()


# RESULTS PAGE
def show_results():
    features  = calculate_features(st.session_state.answers)
    result    = predict_score(features)

    if result[0] is None:
        st.error(
            "Error: models not loaded. Ensure rf_model_tuned.pkl, "
            "lgbm_nomono_model.pkl, and ensemble_config.pkl are present."
        )
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Back to Home", use_container_width=True):
                go_to_home(); st.rerun()
        return

    score, confidence, probability, breakdown = result
    interpretation = get_score_interpretation(score, breakdown)
    insights       = generate_categorized_insights(features, score, breakdown)

    st.markdown(f"""
    <div class="score-container">
        <div class="score-label">Your Behavioral Credit Score</div>
        <div class="score-value" style="color: {interpretation['color']}">{score}</div>
        <div class="score-category" style="color: {interpretation['color']}">{interpretation['category']}</div>
        <div class="score-description">{interpretation['description']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metrics-box">
        <div class="metrics-row">
            <div class="metric-item">
                <div class="metric-label">Model Confidence</div>
                <div class="metric-value">{confidence:.1f}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">
        <h3 class="section-title">Credit Score Ranges</h3>
    </div>
    """, unsafe_allow_html=True)

    ranges = [
        ("Poor",      "300-599", "#c1121f"),
        ("Doubtful",  "600-649", "#f97316"),
        ("Fair",      "650-699", "#f59e0b"),
        ("Good",      "700-749", "#3b82f6"),
        ("Excellent", "750-850", "#2d6a4f"),
    ]
    cols = st.columns(5)
    for col, (cat, rng, color) in zip(cols, ranges):
        with col:
            is_active = interpretation['category'] == cat
            bg    = color if is_active else '#f8f9fa'
            text  = 'white' if is_active else '#1a1a2e'
            fw    = '800' if is_active else '600'
            st.markdown(f"""
            <div class="score-range" style="background:{bg};color:{text};font-weight:{fw};">
                <div style="font-size:1rem;margin-bottom:.25rem;">{cat}</div>
                <div style="font-size:.85rem;">{rng}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='spacer-medium'></div>", unsafe_allow_html=True)

    if score < 700:
        insight_order = [
            ('concerns',  'Areas of Concern',      'concerns'),
            ('actions',   'Recommended Actions',   'actions'),
            ('strengths', 'Your Strengths',         'strengths'),
        ]
    else:
        insight_order = [
            ('strengths', 'Your Strengths',         'strengths'),
            ('concerns',  'Areas of Concern',       'concerns'),
            ('actions',   'Recommended Actions',    'actions'),
        ]

    for key, title, css in insight_order:
        items = insights[key]
        if items:
            st.markdown(f"""
            <div class="insight-category">
                <div class="insight-header {css}">
                    <h3 class="insight-category-title">{title}</h3>
                </div>
                <div class="insight-body">
            """, unsafe_allow_html=True)
            for item in items:
                st.markdown(f"""
                <div class="insight-item {key[:-1]}">
                    <div class="insight-item-title">{item['title']}</div>
                    <div class="insight-item-message">{item['message']}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='spacer-medium'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Calculate Again", use_container_width=True):
            reset_assessment(); st.rerun()
    with col2:
        if st.button("Talk To Financial Assistant", type="primary", use_container_width=True):
            toggle_chatbot(); st.rerun()
    with col3:
        if st.button("Back to Home", use_container_width=True):
            go_to_home(); st.rerun()


# CHATBOT PAGE
def show_chatbot():
    st.markdown("""
    <div class="section-card">
        <h2 class="section-title">Financial Advisor Chatbot</h2>
        <p class="section-description">
            Ask questions about credit scores, budgeting, or financial management
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Clear Chat", use_container_width=True):
            clear_chat(); st.rerun()

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
        toggle_chatbot(); st.rerun()


def main():
    if   st.session_state.page == 'home':       show_home()
    elif st.session_state.page == 'assessment': show_assessment()

if __name__ == "__main__":
    main()