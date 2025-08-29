import streamlit as st

st.set_page_config(page_title="ML Platform", layout="wide")

st.title("🧑‍💻 Machine Learning Platform")

st.write("""
Welcome to your **all-in-one ML platform**! 🚀  
- 📊 Page 1: Upload dataset / Fetch from MySQL, perform EDA & Feature Selection  
- 📈 Page 2: Visualize your data with dynamic plots  
- ⚙️ Page 3: Train ML Models, Evaluate & Tune Hyperparameters  
""")
st.info("👉 Use the sidebar to navigate between pages.")