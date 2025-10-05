"""
Loan Calculator Page
Interactive loan EMI calculator with visualization
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Loan Calculator", page_icon="🧮", layout="wide")

st.markdown("# 🧮 Loan EMI Calculator")
st.markdown("Calculate your monthly payments and visualize loan amortization")

# Input section
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Loan Details")
    loan_amount = st.number_input("Loan Amount ($)", 
                                min_value=1000, 
                                max_value=1000000, 
                                value=100000, 
                                step=5000)
    
    interest_rate = st.slider("Annual Interest Rate (%)", 
                            min_value=1.0, 
                            max_value=20.0, 
                            value=7.5, 
                            step=0.1)
    
    loan_term_years = st.slider("Loan Term (Years)", 
                               min_value=1, 
                               max_value=30, 
                               value=10, 
                               step=1)

with col2:
    st.markdown("### Additional Options")
    
    extra_payment = st.number_input("Extra Monthly Payment ($)", 
                                  min_value=0, 
                                  max_value=10000, 
                                  value=0, 
                                  step=100,
                                  help="Additional payment towards principal")
    
    start_month = st.selectbox("Start Month", 
                              ["January", "February", "March", "April", "May", "June",
                               "July", "August", "September", "October", "November", "December"])
    
    start_year = st.number_input("Start Year", 
                               min_value=2024, 
                               max_value=2030, 
                               value=2024)

# Calculate EMI
def calculate_emi(principal, rate, months):
    """Calculate Equated Monthly Installment (EMI)"""
    monthly_rate = rate / (12 * 100)
    if monthly_rate == 0:
        return principal / months
    emi = principal * monthly_rate * pow(1 + monthly_rate, months) / (pow(1 + monthly_rate, months) - 1)
    return emi

# Calculate amortization schedule
def calculate_amortization(principal, rate, months, extra_payment=0):
    """Calculate detailed amortization schedule"""
    monthly_rate = rate / (12 * 100)
    emi = calculate_emi(principal, rate, months)
    
    balance = principal
    schedule = []
    
    for month in range(1, months + 1):
        interest_payment = balance * monthly_rate
        principal_payment = emi - interest_payment + extra_payment
        
        # Ensure we don't overpay
        if principal_payment > balance:
            principal_payment = balance
        
        balance -= principal_payment
        
        schedule.append({
            'Month': month,
            'EMI': emi,
            'Principal': principal_payment,
            'Interest': interest_payment,
            'Extra Payment': extra_payment if balance > 0 else 0,
            'Balance': max(0, balance)
        })
        
        if balance <= 0:
            break
    
    return pd.DataFrame(schedule)

# Calculations
loan_term_months = loan_term_years * 12
emi = calculate_emi(loan_amount, interest_rate, loan_term_months)
total_payment = emi * loan_term_months + extra_payment * loan_term_months
total_interest = total_payment - loan_amount

# Display results
st.markdown("---")
st.markdown("## 📊 Loan Summary")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric("Monthly EMI", f"${emi:,.2f}")

with metric_col2:
    st.metric("Total Payment", f"${total_payment:,.2f}")

with metric_col3:
    st.metric("Total Interest", f"${total_interest:,.2f}")

with metric_col4:
    interest_percentage = (total_interest / loan_amount) * 100
    st.metric("Interest %", f"{interest_percentage:.1f}%")

# Calculate amortization with and without extra payment
amortization_regular = calculate_amortization(loan_amount, interest_rate, loan_term_months, 0)
amortization_extra = calculate_amortization(loan_amount, interest_rate, loan_term_months, extra_payment)

# Visualizations
tab1, tab2, tab3, tab4 = st.tabs(["Payment Breakdown", "Amortization Chart", 
                                   "Comparison", "Payment Schedule"])

with tab1:
    # Pie chart of total payment breakdown
    fig_pie = go.Figure(data=[go.Pie(
        labels=['Principal', 'Interest'],
        values=[loan_amount, total_interest],
        hole=.4,
        marker_colors=['#1f77b4', '#ff7f0e']
    )])
    fig_pie.update_layout(
        title="Total Payment Breakdown",
        annotations=[dict(text=f'${total_payment:,.0f}', x=0.5, y=0.5, 
                         font_size=20, showarrow=False)]
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    # Line chart showing balance over time
    fig_balance = go.Figure()
    
    fig_balance.add_trace(go.Scatter(
        x=amortization_regular['Month'],
        y=amortization_regular['Balance'],
        mode='lines',
        name='Regular Payment',
        line=dict(color='blue', width=3)
    ))
    
    if extra_payment > 0:
        fig_balance.add_trace(go.Scatter(
            x=amortization_extra['Month'],
            y=amortization_extra['Balance'],
            mode='lines',
            name='With Extra Payment',
            line=dict(color='green', width=3, dash='dash')
        ))
    
    fig_balance.update_layout(
        title='Loan Balance Over Time',
        xaxis_title='Month',
        yaxis_title='Outstanding Balance ($)',
        hovermode='x unified'
    )
    st.plotly_chart(fig_balance, use_container_width=True)
    
    # Stacked area chart for payment breakdown
    fig_payment = go.Figure()
    
    fig_payment.add_trace(go.Scatter(
        x=amortization_regular['Month'],
        y=amortization_regular['Principal'],
        mode='lines',
        name='Principal',
        stackgroup='one',
        fillcolor='rgba(31, 119, 180, 0.5)'
    ))
    
    fig_payment.add_trace(go.Scatter(
        x=amortization_regular['Month'],
        y=amortization_regular['Interest'],
        mode='lines',
        name='Interest',
        stackgroup='one',
        fillcolor='rgba(255, 127, 14, 0.5)'
    ))
    
    fig_payment.update_layout(
        title='Monthly Payment Breakdown',
        xaxis_title='Month',
        yaxis_title='Payment Amount ($)',
        hovermode='x unified'
    )
    st.plotly_chart(fig_payment, use_container_width=True)

with tab3:
    st.markdown("### Impact of Extra Payments")
    
    if extra_payment > 0:
        # Calculate savings
        regular_total = amortization_regular['Interest'].sum()
        extra_total = amortization_extra['Interest'].sum()
        interest_saved = regular_total - extra_total
        months_saved = len(amortization_regular) - len(amortization_extra)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Interest Saved", f"${interest_saved:,.2f}")
        with col2:
            st.metric("Time Saved", f"{months_saved} months")
        with col3:
            years_saved = months_saved / 12
            st.metric("Years Saved", f"{years_saved:.1f} years")
        
        # Comparison bar chart
        comparison_data = pd.DataFrame({
            'Type': ['Regular Payment', 'With Extra Payment'],
            'Total Interest': [regular_total, extra_total],
            'Loan Term (Months)': [len(amortization_regular), len(amortization_extra)]
        })
        
        fig_comp1 = px.bar(comparison_data, x='Type', y='Total Interest',
                          title='Total Interest Comparison',
                          color='Type', color_discrete_map={'Regular Payment': 'red',
                                                          'With Extra Payment': 'green'})
        st.plotly_chart(fig_comp1, use_container_width=True)
    else:
        st.info("Add extra monthly payments to see potential savings!")

with tab4:
    st.markdown("### Detailed Payment Schedule")
    
    # Show first and last 5 payments
    display_df = amortization_regular[['Month', 'EMI', 'Principal', 'Interest', 'Balance']].copy()
    
    # Format currency columns
    currency_cols = ['EMI', 'Principal', 'Interest', 'Balance']
    for col in currency_cols:
        display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}")
    
    # Display options
    view_option = st.radio("View", ["First 12 months", "Last 12 months", "Full schedule"])
    
    if view_option == "First 12 months":
        st.dataframe(display_df.head(12), use_container_width=True)
    elif view_option == "Last 12 months":
        st.dataframe(display_df.tail(12), use_container_width=True)
    else:
        st.dataframe(display_df, use_container_width=True)
    
    # Download button
    csv = amortization_regular.to_csv(index=False)
    st.download_button(
        label="Download Full Schedule as CSV",
        data=csv,
        file_name=f"loan_amortization_schedule_{loan_amount}_{interest_rate}.csv",
        mime="text/csv"
    )

# Additional Information
st.markdown("---")
st.markdown("## 💡 Loan Tips")

tips_col1, tips_col2 = st.columns(2)

with tips_col1:
    st.info("""
    **Ways to Save on Interest:**
    - Make extra payments towards principal
    - Choose bi-weekly payments instead of monthly
    - Refinance when rates drop significantly
    - Make a larger down payment
    """)

with tips_col2:
    st.warning("""
    **Important Considerations:**
    - Check for prepayment penalties
    - Maintain emergency fund before extra payments
    - Consider tax implications
    - Compare multiple lenders
    """)

# Loan comparison tool
st.markdown("---")
st.markdown("## 🔄 Compare Different Loan Options")

compare_col1, compare_col2 = st.columns(2)

with compare_col1:
    st.markdown("#### Option 1")
    amount1 = st.number_input("Amount ($)", min_value=1000, max_value=1000000, 
                            value=loan_amount, key="amount1")
    rate1 = st.number_input("Rate (%)", min_value=1.0, max_value=20.0, 
                          value=interest_rate, key="rate1")
    term1 = st.number_input("Term (Years)", min_value=1, max_value=30, 
                          value=loan_term_years, key="term1")

with compare_col2:
    st.markdown("#### Option 2")
    amount2 = st.number_input("Amount ($)", min_value=1000, max_value=1000000, 
                            value=loan_amount, key="amount2")
    rate2 = st.number_input("Rate (%)", min_value=1.0, max_value=20.0, 
                          value=interest_rate + 0.5, key="rate2")
    term2 = st.number_input("Term (Years)", min_value=1, max_value=30, 
                          value=loan_term_years, key="term2")

if st.button("Compare Options"):
    # Calculate for both options
    emi1 = calculate_emi(amount1, rate1, term1 * 12)
    emi2 = calculate_emi(amount2, rate2, term2 * 12)
    
    total1 = emi1 * term1 * 12
    total2 = emi2 * term2 * 12
    
    interest1 = total1 - amount1
    interest2 = total2 - amount2
    
    # Display comparison
    comparison = pd.DataFrame({
        'Metric': ['Monthly EMI', 'Total Payment', 'Total Interest', 'Interest Rate'],
        'Option 1': [f"${emi1:,.2f}", f"${total1:,.2f}", f"${interest1:,.2f}", f"{rate1}%"],
        'Option 2': [f"${emi2:,.2f}", f"${total2:,.2f}", f"${interest2:,.2f}", f"{rate2}%"]
    })
    
    st.table(comparison)
    
    if total1 < total2:
        savings = total2 - total1
        st.success(f"Option 1 saves ${savings:,.2f} over the loan term!")
    else:
        savings = total1 - total2
        st.success(f"Option 2 saves ${savings:,.2f} over the loan term!")