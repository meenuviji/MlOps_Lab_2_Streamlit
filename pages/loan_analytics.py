"""
Loan Analytics Page
Provides detailed analytics and insights about loan applications
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Loan Analytics", page_icon="📊", layout="wide")

st.markdown("# 📊 Loan Analytics Dashboard")
st.markdown("Comprehensive analytics for loan applications and approvals")

# Generate sample data for demonstration
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    n_samples = 500
    
    # Generate synthetic loan data
    dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='D')
    
    data = pd.DataFrame({
        'date': dates,
        'loan_amount': np.random.normal(150000, 50000, n_samples).clip(10000, 500000),
        'credit_score': np.random.normal(700, 80, n_samples).clip(300, 850),
        'income': np.random.normal(75000, 25000, n_samples).clip(20000, 200000),
        'approval': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'interest_rate': np.random.normal(7.5, 2, n_samples).clip(3, 15),
        'loan_term': np.random.choice([60, 120, 180, 240, 360], n_samples),
        'purpose': np.random.choice(['Home', 'Auto', 'Personal', 'Business', 'Education'], n_samples),
        'state': np.random.choice(['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA'], n_samples)
    })
    
    # Adjust approval based on credit score
    data.loc[data['credit_score'] < 650, 'approval'] = np.random.choice([0, 1], 
                                                                       sum(data['credit_score'] < 650), 
                                                                       p=[0.8, 0.2])
    data.loc[data['credit_score'] > 750, 'approval'] = np.random.choice([0, 1], 
                                                                       sum(data['credit_score'] > 750), 
                                                                       p=[0.1, 0.9])
    
    return data

# Load data
df = generate_sample_data()

# Key Metrics
st.markdown("## 📈 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

with col1:
    approval_rate = df['approval'].mean() * 100
    st.metric("Approval Rate", f"{approval_rate:.1f}%", 
              delta=f"{approval_rate - 65:.1f}% vs target")

with col2:
    avg_loan_amount = df['loan_amount'].mean()
    st.metric("Avg Loan Amount", f"${avg_loan_amount:,.0f}", 
              delta=f"${avg_loan_amount - 140000:,.0f}")

with col3:
    avg_credit_score = df[df['approval'] == 1]['credit_score'].mean()
    st.metric("Avg Credit Score (Approved)", f"{avg_credit_score:.0f}")

with col4:
    total_volume = df['loan_amount'].sum() / 1e6
    st.metric("Total Volume", f"${total_volume:.1f}M")

# Time Series Analysis
st.markdown("## 📅 Trends Over Time")

tab1, tab2, tab3 = st.tabs(["Approval Trends", "Volume Analysis", "Credit Score Distribution"])

with tab1:
    # Approval rate over time
    daily_approvals = df.groupby(df['date'].dt.to_period('W'))['approval'].agg(['mean', 'count'])
    daily_approvals.index = daily_approvals.index.to_timestamp()
    
    fig_approval = go.Figure()
    fig_approval.add_trace(go.Scatter(
        x=daily_approvals.index,
        y=daily_approvals['mean'] * 100,
        mode='lines+markers',
        name='Approval Rate',
        line=dict(color='green', width=3)
    ))
    fig_approval.update_layout(
        title='Weekly Approval Rate Trend',
        xaxis_title='Week',
        yaxis_title='Approval Rate (%)',
        hovermode='x unified'
    )
    st.plotly_chart(fig_approval, use_container_width=True)

with tab2:
    # Loan volume analysis
    volume_by_purpose = df.groupby(['purpose', 'approval'])['loan_amount'].sum().reset_index()
    volume_by_purpose['status'] = volume_by_purpose['approval'].map({0: 'Rejected', 1: 'Approved'})
    
    fig_volume = px.bar(volume_by_purpose, x='purpose', y='loan_amount', 
                        color='status', title='Loan Volume by Purpose and Status',
                        labels={'loan_amount': 'Total Volume ($)', 'purpose': 'Loan Purpose'},
                        color_discrete_map={'Approved': '#28a745', 'Rejected': '#dc3545'})
    st.plotly_chart(fig_volume, use_container_width=True)

with tab3:
    # Credit score distribution
    fig_credit = go.Figure()
    
    approved_scores = df[df['approval'] == 1]['credit_score']
    rejected_scores = df[df['approval'] == 0]['credit_score']
    
    fig_credit.add_trace(go.Histogram(
        x=approved_scores,
        name='Approved',
        opacity=0.7,
        marker_color='green'
    ))
    fig_credit.add_trace(go.Histogram(
        x=rejected_scores,
        name='Rejected',
        opacity=0.7,
        marker_color='red'
    ))
    fig_credit.update_layout(
        title='Credit Score Distribution by Approval Status',
        xaxis_title='Credit Score',
        yaxis_title='Count',
        barmode='overlay'
    )
    st.plotly_chart(fig_credit, use_container_width=True)

# Risk Analysis
st.markdown("## 🎯 Risk Analysis")

col1, col2 = st.columns(2)

with col1:
    # Approval rate by credit score range
    credit_ranges = pd.cut(df['credit_score'], 
                          bins=[300, 580, 650, 700, 750, 850],
                          labels=['Very Poor', 'Poor', 'Fair', 'Good', 'Excellent'])
    approval_by_credit = df.groupby(credit_ranges)['approval'].mean() * 100
    
    fig_risk1 = px.bar(x=approval_by_credit.index, y=approval_by_credit.values,
                       title='Approval Rate by Credit Score Range',
                       labels={'x': 'Credit Score Range', 'y': 'Approval Rate (%)'},
                       color=approval_by_credit.values,
                       color_continuous_scale='RdYlGn')
    st.plotly_chart(fig_risk1, use_container_width=True)

with col2:
    # Scatter plot: Income vs Loan Amount
    fig_risk2 = px.scatter(df, x='income', y='loan_amount', 
                          color='approval', 
                          title='Income vs Loan Amount',
                          labels={'approval': 'Status'},
                          color_discrete_map={0: 'red', 1: 'green'},
                          opacity=0.6)
    fig_risk2.update_traces(marker=dict(size=8))
    st.plotly_chart(fig_risk2, use_container_width=True)

# Geographic Analysis
st.markdown("## 🗺️ Geographic Distribution")

state_stats = df.groupby('state').agg({
    'approval': 'mean',
    'loan_amount': 'sum',
    'credit_score': 'mean'
}).reset_index()

col1, col2 = st.columns(2)

with col1:
    fig_geo1 = px.bar(state_stats.sort_values('approval', ascending=False), 
                      x='state', y='approval',
                      title='Approval Rate by State',
                      labels={'approval': 'Approval Rate'},
                      color='approval',
                      color_continuous_scale='Viridis')
    fig_geo1.update_yaxis(tickformat='.0%')
    st.plotly_chart(fig_geo1, use_container_width=True)

with col2:
    fig_geo2 = px.pie(state_stats, values='loan_amount', names='state',
                      title='Loan Volume Distribution by State')
    st.plotly_chart(fig_geo2, use_container_width=True)

# Predictive Insights
st.markdown("## 🔮 Predictive Insights")

insight_col1, insight_col2 = st.columns(2)

with insight_col1:
    st.info("""
    **🎯 High Approval Factors:**
    - Credit Score > 700 (85% approval rate)
    - Income > $60,000 (78% approval rate)
    - Debt-to-Income < 40% (82% approval rate)
    - Employment > 2 years (75% approval rate)
    """)

with insight_col2:
    st.warning("""
    **⚠️ Risk Indicators:**
    - Credit Score < 650 (20% approval rate)
    - High loan-to-income ratio (>5x)
    - Previous defaults or bankruptcies
    - Unstable employment history
    """)

# Download section
st.markdown("## 📥 Export Data")

if st.button("Generate Report"):
    # Create a summary report
    report = f"""
    Loan Analytics Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    Key Metrics:
    - Total Applications: {len(df)}
    - Approval Rate: {approval_rate:.1f}%
    - Average Loan Amount: ${avg_loan_amount:,.0f}
    - Average Credit Score (Approved): {avg_credit_score:.0f}
    
    Top Performing States:
    {state_stats.sort_values('approval', ascending=False).head(3).to_string()}
    """
    
    st.download_button(
        label="Download Report",
        data=report,
        file_name=f"loan_analytics_report_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain"
    )