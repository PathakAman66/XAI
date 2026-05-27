import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="XAI Demo",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

html, body, [class*="css"] {
    font-size: 15px;
}

h1, h2, h3 {
    color: #00FFAA;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

</style>
""", unsafe_allow_html=True)

st.title("🤖 Before XAI vs After XAI")

st.write("### Transparent and Trustworthy AI Demonstration")

data = pd.DataFrame({
    'Income': [20000, 50000, 80000, 120000],
    'CreditScore': [300, 600, 700, 800],
    'Age': [22, 35, 45, 50],
    'Approved': [0, 0, 1, 1]
})

X = data[['Income', 'CreditScore', 'Age']]
y = data['Approved']

model = RandomForestClassifier()
model.fit(X, y)

st.sidebar.header("Applicant Details")

income = st.sidebar.slider("Income", 10000, 150000, 30000)
credit = st.sidebar.slider("Credit Score", 300, 900, 500)
age = st.sidebar.slider("Age", 18, 60, 24)

input_data = pd.DataFrame(
    [[income, credit, age]],
    columns=['Income', 'CreditScore', 'Age']
)

prediction = model.predict(input_data)

col1, col2 = st.columns(2)

with col1:

    st.subheader("❌ BEFORE XAI")

    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")

    st.write("### Problem:")
    st.write("- No explanation provided")
    st.write("- User cannot understand decision")
    st.write("- Low transparency")
    st.write("- Difficult to trust AI system")
    

with col2:

    st.subheader("✅ AFTER XAI")

    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")

  
    importances = model.feature_importances_
    features = X.columns

    fig, ax = plt.subplots(figsize=(4,2))

    ax.bar(features, importances)

    ax.set_title("Feature Importance")

    st.pyplot(fig)


    reason = ""

    if credit < 600:
        reason += "Low credit score. "

    if income < 40000:
        reason += "Low income. "

    if age < 25:
        reason += "Limited financial history."

    if reason == "":
        reason = "Good financial profile."

    st.info(reason)


st.success("✅ Explainable AI improves transparency and trust.")