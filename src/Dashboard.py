"""
Loan Approval Prediction Dashboard - Fixed Version Without File Check
This version removes the unnecessary model file check
"""

import json
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from streamlit.logger import get_logger
from datetime import datetime

# THIS MUST BE FIRST - Before any other st. commands!
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="wide"
)

# Configuration
FASTAPI_BACKEND_ENDPOINT = "http://localhost:8000"
# We don't need to check for model file since API handles it
# FASTAPI_LOAN_MODEL_LOCATION = Path(__file__).resolve().parents[2] / 'FastAPI_Labs' / 'src' / 'loan_model.pkl'
LOGGER = get_logger(__name__)

# Custom CSS
st.markdown("""
<style>
    .loan-approved { 
        background-color: #d4edda; 
        color: #155724;
        padding: 20px; 
        border-radius: 10px; 
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .loan-rejected { 
        background-color: #f8d7da; 
        color: #721c24;
        padding: 20px; 
        border-radius: 10px; 
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def calculate_loan_metrics(loan_amount, interest_rate, loan_term):
    """Calculate EMI and total payment"""
    monthly_rate = interest_rate / (12 * 100)
    emi = loan_amount * monthly_rate * pow(1 + monthly_rate, loan_term) / (pow(1 + monthly_rate, loan_term) - 1)
    total_payment = emi * loan_term
    total_interest = total_payment - loan_amount
    return emi, total_payment, total_interest

def create_gauge_chart(probability, title="Approval Probability"):
    """Create a gauge chart for approval probability"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "#ff4444"},
                {'range': [30, 70], 'color': "#ffaa00"},
                {'range': [70, 100], 'color': "#00ff00"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def run():
    # Initialize session state
    if 'prediction_history' not in st.session_state:
        st.session_state['prediction_history'] = []
    
    with st.sidebar:
        st.markdown("# 💰 Loan Application Portal")
        
        # Backend status check
        backend_status = False
        try:
            backend_request = requests.get(f"{FASTAPI_BACKEND_ENDPOINT}/health")
            if backend_request.status_code == 200:
                health_data = backend_request.json()
                if health_data.get("model_loaded", False):
                    st.success("✅ Backend System Online")
                    backend_status = True
                else:
                    st.warning("⚠️ Backend online but model not loaded")
            else:
                st.error("❌ Backend connection issue")
        except requests.ConnectionError as ce:
            LOGGER.error(ce)
            st.error("❌ Backend System Offline")
            st.info("Please start the FastAPI server:\n```\nuvicorn main_loan:app --reload\n```")
        
        st.divider()
        
        # Input method selection
        input_method = st.radio(
            "Application Method:",
            ["📝 Manual Application", "📁 Upload Application", "🎲 Sample Application"]
        )
        
        st.info("Enter loan application details")
        
        application_data = None
        loan_amount = 100000
        loan_term = 120
        interest_rate = 7.5
        
        if input_method == "📝 Manual Application":
            # Personal Information
            st.markdown("### Personal Information")
            age = st.slider("Age", 18, 75, 35, 1, help="Applicant age in years")
            income = st.number_input("Annual Income ($)", 20000, 500000, 50000, 5000, 
                                   help="Total annual income before taxes")
            
            # Loan Details
            st.markdown("### Loan Details")
            loan_amount = st.number_input("Loan Amount ($)", 1000, 1000000, 100000, 5000)
            loan_term = st.slider("Loan Term (months)", 12, 360, 120, 12, 
                                help="Loan duration in months")
            interest_rate = st.slider("Interest Rate (%)", 3.0, 20.0, 7.5, 0.5)
            
            # Credit Information
            st.markdown("### Credit Information")
            credit_score = st.slider("Credit Score", 300, 850, 650, 10,
                                   help="FICO credit score")
            
            credit_help = """
            - Excellent: 750-850
            - Good: 700-749  
            - Fair: 650-699
            - Poor: Below 650
            """
            with st.expander("Credit Score Guide"):
                st.markdown(credit_help)
            
            employment_length = st.slider("Employment Length (years)", 0, 40, 5, 1,
                                        help="Years at current job")
            
            # Additional Features
            col1, col2 = st.columns(2)
            with col1:
                home_ownership = st.selectbox("Home Ownership", 
                                            ["Rent", "Own", "Mortgage"])
            with col2:
                purpose = st.selectbox("Loan Purpose",
                                     ["Debt Consolidation", "Home Improvement", 
                                      "Business", "Education", "Other"])
            
            has_cosigner = st.checkbox("Has Co-signer")
            previous_defaults = st.checkbox("Previous Defaults")
            
            # Calculate debt-to-income ratio
            monthly_debt = st.number_input("Monthly Debt Payments ($)", 0, 10000, 500, 100)
            dti_ratio = (monthly_debt * 12) / income if income > 0 else 0
            
            # Prepare data
            application_data = {
                "age": age,
                "income": income,
                "loan_amount": loan_amount,
                "loan_term": loan_term,
                "credit_score": credit_score,
                "employment_length": employment_length,
                "home_ownership": {"Rent": 0, "Own": 1, "Mortgage": 2}[home_ownership],
                "purpose": {"Debt Consolidation": 0, "Home Improvement": 1, 
                          "Business": 2, "Education": 3, "Other": 4}[purpose],
                "dti_ratio": round(dti_ratio, 2),
                "has_cosigner": int(has_cosigner),
                "previous_defaults": int(previous_defaults)
            }
            
        elif input_method == "📁 Upload Application":
            test_input_file = st.file_uploader('Upload loan application', type=['json'])
            if test_input_file:
                st.write('📄 Application Preview')
                test_input_data = json.load(test_input_file)
                st.json(test_input_data)
                application_data = test_input_data.get("input_test", test_input_data)
                
                # Extract loan details for calculation
                loan_amount = application_data.get("loan_amount", 100000)
                loan_term = application_data.get("loan_term", 120)
                interest_rate = 7.5  # Default interest rate
                
        else:  # Sample Application
            if st.button("Generate Sample Application"):
                application_data = {
                    "age": 35,
                    "income": 75000,
                    "loan_amount": 150000,
                    "loan_term": 180,
                    "credit_score": 720,
                    "employment_length": 8,
                    "home_ownership": 1,  # Own
                    "purpose": 0,  # Debt Consolidation
                    "dti_ratio": 0.25,
                    "has_cosigner": 0,
                    "previous_defaults": 0
                }
                st.json(application_data)
                loan_amount = 150000
                loan_term = 180
                interest_rate = 7.5
        
        st.divider()
        predict_button = st.button('🔍 Analyze Application', type="primary", 
                                 use_container_width=True,
                                 disabled=not backend_status)
    
    # Main content area
    st.write("# Loan Approval Prediction System 💰")
    st.markdown("### AI-Powered Instant Loan Decision")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Application Analysis", "📊 Risk Assessment", 
                                     "📈 History", "ℹ️ Information"])
    
    with tab1:
        if predict_button and application_data:
            # REMOVED FILE CHECK - Just use API directly
            result_container = st.empty()
            
            with st.spinner('🔍 Analyzing loan application...'):
                try:
                    # Make prediction request
                    predict_response = requests.post(
                        f'{FASTAPI_BACKEND_ENDPOINT}/predict',
                        json=application_data
                    )
                    
                    if predict_response.status_code == 200:
                        result = predict_response.json()
                        
                        # Display main result
                        if result.get("response", result.get("approval_status") == "Approved") == 1:
                            st.markdown('<div class="loan-approved">✅ LOAN APPROVED</div>', 
                                      unsafe_allow_html=True)
                            st.balloons()
                        else:
                            st.markdown('<div class="loan-rejected">❌ LOAN REJECTED</div>', 
                                      unsafe_allow_html=True)
                        
                        # Display approval probability (if available)
                        if "probability" in result:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.plotly_chart(create_gauge_chart(result["probability"]), 
                                              use_container_width=True)
                            with col2:
                                st.metric("Approval Probability", 
                                        f"{result['probability']*100:.1f}%")
                                if "confidence" in result:
                                    st.metric("Confidence Level", result['confidence'])
                        
                        # Risk factors and recommendations
                        if "risk_factors" in result and result["risk_factors"]:
                            st.markdown("### ⚠️ Risk Factors")
                            for factor in result["risk_factors"]:
                                st.warning(f"• {factor}")
                        
                        if "recommendations" in result and result["recommendations"]:
                            st.markdown("### 💡 Recommendations")
                            for rec in result["recommendations"]:
                                if result.get("response", 0) == 1:
                                    st.success(f"• {rec}")
                                else:
                                    st.info(f"• {rec}")
                        
                        # Loan details calculation
                        if 'loan_amount' in application_data:
                            st.markdown("### 💵 Loan Details")
                            emi, total_payment, total_interest = calculate_loan_metrics(
                                application_data['loan_amount'],
                                interest_rate,
                                application_data['loan_term']
                            )
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Loan Amount", f"${application_data['loan_amount']:,.0f}")
                            with col2:
                                st.metric("Monthly EMI", f"${emi:,.2f}")
                            with col3:
                                st.metric("Total Interest", f"${total_interest:,.2f}")
                            with col4:
                                st.metric("Total Payment", f"${total_payment:,.2f}")
                        
                        # Add to history
                        st.session_state.prediction_history.append({
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'application': application_data,
                            'result': result.get("approval_status", "Approved" if result.get("response") == 1 else "Rejected")
                        })
                        
                    else:
                        st.error(f"Prediction failed: Status {predict_response.status_code}")
                        st.error(f"Response: {predict_response.text}")
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"API Error: {str(e)}")
                    st.error("Please make sure the FastAPI backend is running on port 8000")
                    LOGGER.error(f"API error: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")
                    LOGGER.error(f"Prediction error: {e}")
                    
        elif predict_button and not application_data:
            st.warning("Please provide application data before analyzing!")
        elif not backend_status:
            st.error("Cannot make predictions - Backend system is offline!")
    
    with tab2:
        st.markdown("### 📊 Risk Factors Analysis")
        
        # Risk factors visualization
        risk_factors = pd.DataFrame({
            'Factor': ['Credit Score', 'Income Level', 'DTI Ratio', 'Employment', 'Loan Amount'],
            'Impact': [35, 25, 20, 15, 5],
            'Status': ['Good', 'Excellent', 'Fair', 'Good', 'Moderate']
        })
        
        fig = px.bar(risk_factors, x='Impact', y='Factor', orientation='h',
                     color='Status', title='Risk Factor Importance',
                     color_discrete_map={'Excellent': '#28a745', 'Good': '#17a2b8', 
                                       'Fair': '#ffc107', 'Moderate': '#6c757d'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Credit score ranges
        st.markdown("### 📈 Credit Score Guidelines")
        col1, col2 = st.columns(2)
        with col1:
            st.info("""
            **Credit Score Ranges:**
            - 🌟 Excellent: 750-850 (Best rates)
            - ✅ Good: 700-749 (Good rates)
            - ⚠️ Fair: 650-699 (Higher rates)
            - ❌ Poor: Below 650 (May be rejected)
            """)
        with col2:
            st.info("""
            **DTI Ratio Guidelines:**
            - 🌟 Excellent: Below 20%
            - ✅ Good: 20-35%
            - ⚠️ Fair: 36-49%
            - ❌ Poor: 50% or above
            """)
    
    with tab3:
        st.markdown("### 📜 Application History")
        if st.session_state.prediction_history:
            history_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(history_df, use_container_width=True)
            
            # Summary statistics
            approved_count = sum(1 for h in st.session_state.prediction_history 
                               if h['result'] == 'Approved')
            total_count = len(st.session_state.prediction_history)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Applications", total_count)
            with col2:
                st.metric("Approved", approved_count)
            with col3:
                approval_rate = (approved_count / total_count * 100) if total_count > 0 else 0
                st.metric("Approval Rate", f"{approval_rate:.1f}%")
        else:
            st.info("No applications analyzed yet. Submit an application to see history.")
    
    with tab4:
        st.markdown("### ℹ️ About This System")
        st.info("""
        **Loan Approval Prediction System**
        
        This AI-powered system uses machine learning to provide instant loan approval decisions 
        based on multiple factors including credit score, income, debt-to-income ratio, and more.
        
        **Features:**
        - Instant approval decision
        - Risk assessment and scoring  
        - EMI calculation
        - Application history tracking
        - Detailed recommendations
        
        **Model Information:**
        - Algorithm: Random Forest Classification
        - Accuracy: ~92% on test data
        - Features: 11 key financial indicators
        
        **Created for:** MLOps Lab Assignment
        **Modified by:** [Your Name]
        
        **Disclaimer:** This is a demo system for educational purposes. Actual loan approvals 
        require additional verification and are subject to bank policies.
        """)

if __name__ == "__main__":
    run()