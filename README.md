# Loan Approval Prediction System 💰

## MLOps Lab Assignment - Streamlit Dashboard Modification

This project is a comprehensive modification of the original Iris flower classification Streamlit lab, transformed into a practical **Loan Approval Prediction System** using machine learning and modern web technologies.

## 🚀 Overview

The Loan Approval Prediction System is an end-to-end MLOps solution that predicts loan approval decisions based on applicant financial data. It features a user-friendly Streamlit interface connected to a FastAPI backend with a trained Random Forest model.

### Original Lab vs. Modified System

| Aspect | Original Lab | Modified System |
|--------|--------------|-----------------|
| **Use Case** | Iris flower classification | Loan approval prediction |
| **Model Type** | 3-class classification | Binary classification (Approved/Rejected) |
| **Input Features** | 4 flower measurements | 11 financial indicators |
| **UI Design** | Basic single page | Multi-page application with analytics |
| **Visualizations** | Simple output display | Interactive Plotly charts & gauges |
| **User Input** | Manual sliders only | Manual, file upload, and sample generation |
| **Output** | Flower type | Detailed risk analysis with recommendations |

## 🎯 Key Features

### 1. **Enhanced Dashboard** (`Dashboard.py`)
- **Real-time Predictions**: Instant loan approval decisions with confidence scores
- **Multiple Input Methods**:
  - 📝 Manual entry with intuitive sliders and forms
  - 📁 JSON file upload for bulk processing
  - 🎲 Sample application generator for testing
- **Rich Visualizations**: 
  - Gauge charts for approval probability
  - Risk factor analysis
  - EMI calculations with payment breakdowns
- **Smart Recommendations**: Personalized advice based on approval status

### 2. **Multi-Page Architecture**
- **📊 Loan Analytics** (`pages/Loan_Analytics.py`)
  - KPI dashboard with approval trends
  - Geographic distribution analysis
  - Risk factor importance visualization
  - Export functionality for reports
  
- **🧮 Loan Calculator** (`pages/Loan_Calculator.py`)
  - Interactive EMI calculator
  - Amortization schedule generator
  - Extra payment impact analysis
  - Loan comparison tool

### 3. **Advanced ML Features**
- **11 Input Features**:
  1. Age (18-75 years)
  2. Annual Income ($20k-$500k)
  3. Loan Amount ($1k-$1M)
  4. Loan Term (12-360 months)
  5. Credit Score (300-850)
  6. Employment Length
  7. Home Ownership Status
  8. Loan Purpose
  9. Debt-to-Income Ratio
  10. Co-signer Status
  11. Previous Default History

### 4. **Professional UI/UX**
- Custom CSS styling for better aesthetics
- Color-coded approval/rejection displays
- Session-based history tracking
- Responsive layout with tabs and columns
- Loading spinners and progress indicators

## 🛠️ Technical Stack

- **Frontend**: Streamlit 1.28.0
- **Backend**: FastAPI 0.104.1
- **ML Model**: Scikit-learn Random Forest Classifier
- **Visualization**: Plotly 5.18.0
- **Data Processing**: Pandas, NumPy
- **Model Accuracy**: ~92% on test data

## 📦 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### 1. Clone the Repository
```bash
git clone [your-repo-url]
cd Lab2
```

### 2. Install Dependencies
```bash
# Navigate to Streamlit Labs
cd Streamlit_Labs

# Create virtual environment
python -m venv loanenv

# Activate environment
source loanenv/bin/activate  # Mac/Linux
# or
loanenv\Scripts\activate  # Windows

# Install packages
pip install -r requirements.txt
```

### 3. Train the Model
```bash
# Navigate to FastAPI source
cd ../FastAPI_Labs/src/

# Run training script
python train_loan_model.py
```

### 4. Start the Servers

**Terminal 1 - FastAPI Backend:**
```bash
cd FastAPI_Labs/src/
uvicorn main_loan:app --reload
```

**Terminal 2 - Streamlit Frontend:**
```bash
cd Streamlit_Labs/src/
streamlit run Dashboard.py
```

### 5. Access the Application
- Streamlit Dashboard: http://localhost:8501
- FastAPI Documentation: http://localhost:8000/docs

## 📊 Usage Guide

### Making Predictions

1. **File Upload Method**:
   - Select "📁 Upload Application"
   - Upload a JSON file with loan application data
   - Click "🔍 Analyze Application"

2. **Manual Input Method**:
   - Select "📝 Manual Application"
   - Adjust sliders for each parameter
   - Click "🔍 Analyze Application"

3. **Sample Application**:
   - Select "🎲 Sample Application"
   - Click "Generate Sample Application"
   - Click "🔍 Analyze Application"

### Understanding Results

- **✅ LOAN APPROVED**: Application meets criteria with recommendations for maintaining good standing
- **❌ LOAN REJECTED**: Application needs improvement with specific actionable advice
- **Risk Factors**: Detailed breakdown of factors affecting the decision
- **Confidence Score**: Model's confidence in the prediction

### Sample Test Data

Example `test_loan.json`:
```json
{
  "input_test": {
    "age": 35,
    "income": 75000,
    "loan_amount": 200000,
    "loan_term": 240,
    "credit_score": 720,
    "employment_length": 5,
    "home_ownership": 1,
    "purpose": 0,
    "dti_ratio": 0.35,
    "has_cosigner": 0,
    "previous_defaults": 0
  }
}
```

## 🔧 Project Structure

```
Lab2/
├── Streamlit_Labs/
│   ├── src/
│   │   ├── Dashboard.py          # Main application
│   │   ├── pages/               # Multi-page apps
│   │   │   ├── 1_📊_Loan_Analytics.py
│   │   │   └── 2_🧮_Loan_Calculator.py
│   │   └── test_loan.json       # Sample test data
│   └── requirements.txt
│
└── FastAPI_Labs/
    └── src/
        ├── main_loan.py         # API endpoints
        ├── train_loan_model.py  # Model training
        ├── loan_model.pkl       # Trained model
        └── loan_scaler.pkl      # Feature scaler
```

## 📈 Model Performance

- **Algorithm**: Random Forest Classifier (100 estimators)
- **Accuracy**: ~92% on test set
- **Training Samples**: 5,000 synthetic loan applications
- **Feature Importance**: Credit Score (35%), Income (25%), DTI Ratio (20%)

## 🎨 UI Enhancements

### Custom Styling
- Professional color scheme with green/red indicators
- Responsive layout adapting to screen sizes
- Interactive Plotly charts with hover details
- Custom CSS for loan approval/rejection displays

### User Experience
- Real-time backend status monitoring
- Clear error messages with troubleshooting steps
- Session-based history tracking
- Intuitive navigation with descriptive icons

## 🔄 Key Modifications from Original Lab

1. **Complete Domain Change**: From botanical classification to financial services
2. **Enhanced Complexity**: 11 features vs 4, with real-world business logic
3. **Multi-Page Structure**: Added analytics and calculator pages
4. **Rich Visualizations**: Interactive charts replacing simple text output
5. **Practical Features**: EMI calculation, risk assessment, recommendations
6. **Professional Design**: Custom styling and responsive layout
7. **Extended API**: Additional endpoints for health checks and model info

## 🚀 Future Enhancements

- Database integration for persistent storage
- User authentication and role-based access
- Real-time model retraining pipeline
- PDF report generation
- Email notifications for applications
- A/B testing framework for model versions
- Integration with external credit bureaus

## 📝 Assignment Notes

This project demonstrates:
- Understanding of MLOps concepts and pipeline architecture
- Ability to transform and enhance existing codebases
- Implementation of practical, real-world applications
- Integration of multiple technologies (Streamlit, FastAPI, ML)
- Focus on user experience and professional design

## 👨‍💻 Author

**[Your Name]**  
MLOps Lab Assignment - Fall 2024  
Northeastern University

## 📄 License

This project is for educational purposes as part of the MLOps course at Northeastern University.

## 🙏 Acknowledgments

- Original Streamlit lab structure from the MLOps course repository
- FastAPI framework for robust API development
- Scikit-learn for machine learning capabilities
- Plotly for interactive visualizations