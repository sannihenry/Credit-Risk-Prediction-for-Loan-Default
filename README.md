**💳 Credit Risk Prediction System
📌 Overview**

This project builds a machine learning model to predict loan default risk, a critical problem in the banking and financial services industry.
The goal is to help financial institutions identify high-risk borrowers before approving loans, reducing potential losses and improving decision-making.

**🎯 Business Problem**

Loan defaults are a major source of financial loss for banks.
By leveraging historical customer data, this project predicts whether a borrower is likely to default within 2 years, enabling:
Better credit approval decisions
Risk-based pricing
Improved portfolio management
📊 Dataset
Source: Kaggle – Give Me Some Credit
Type: Real-world financial data
Target Variable: SeriousDlqin2yrs (renamed to target)
Key Features:
Revolving credit utilization
Age
Debt ratio
Monthly income
Number of open credit lines
Past delinquency history

**Models used:**
Logistic Regression (baseline)
Random Forest (optional)
XGBoost (advanced)
6. Model Evaluation
**
Metrics used:**
Accuracy
Precision & Recall
F1 Score
ROC-AUC Score
Confusion Matrix
📈 Results
The model successfully identifies high-risk borrowers
ROC-AUC score demonstrates strong classification performance
Key predictors of default include:
Credit utilization
Debt ratio
Past delinquency history
**📂 Project Structure**
credit-risk-project/
│
├── data/
├── notebooks/
├── src/
├── models/
├── app/
├── README.md
**📌 How to Run**
Clone the repository:
git clone https://github.com/your-username/credit-risk-project.git
Install dependencies:
pip install -r requirements.txt
Run the notebook or scripts

👤 Author

Sanni Henry

⭐ If you found this useful

Give this repo a star ⭐
