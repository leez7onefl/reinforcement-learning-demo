import streamlit as st
monitoring_page = st.Page("monitoring.py", title="Monitoring", icon="")
train_page = st.Page("training.py", title="Training", icon="")
test_page = st.Page("evaluation.py", title="Eval", icon="")
pg = st.navigation([monitoring_page, train_page, test_page])
st.set_page_config(page_title="RL Final Project", layout="wide", page_icon="📦")
pg.run()