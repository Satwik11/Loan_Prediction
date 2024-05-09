from os.path import join
import streamlit as st
from PIL import Image
import pickle



def run(model):
    st.title("Bank Loan Prediction")

    grad = st.selectbox('Is Graduated', ['No', 'Yes'])
    graduated = grad == 'Yes'

    self_employed = st.selectbox('Self Employed', ['No', 'Yes']) == 'Yes'

    annual_income = st.number_input("Annual Income($)", value=0)
    loan_amt = st.number_input("Loan Amount($)", value=0)
    loan_term = st.number_input("Loan Term(Months)", min_value=1, value=12)
    civil_score = st.number_input("Credit Score", value=0)
   
    residential_asset = st.number_input("Residential Asset Value($)", value=0)
    bank_asset = st.number_input("Bank Asset Value($)", value=0)

    if st.button("Submit"):
        duration = loan_term

        # Convert categorical features to numerical
        grad_num = 1 if graduated else 0
        emp_status_num = 1 if self_employed else 0

        # Use the model to predict loan approval
        prediction = model.predict([[grad_num, emp_status_num, annual_income, loan_amt, duration, civil_score,
                                      residential_asset, bank_asset]])

        if prediction == 0:
            st.error("According to our calculations, you will not get the loan from the bank.")
        else:
            st.success("Congratulations!! You will get the loan from the bank.")

if __name__ == '__main__':
    model_path = join('model', 'loan_approval_models.pkl')
    rf_model = pickle.load(open(model_path, 'rb'))
    run(rf_model)