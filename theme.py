import streamlit as st
def inject_theme():
    st.markdown("""<style>
.stApp{background:radial-gradient(circle at 10% 0%,rgba(80,160,255,.10),transparent 30%),radial-gradient(circle at 90% 5%,rgba(130,100,255,.08),transparent 30%)}
[data-testid="stMetric"]{border:1px solid rgba(127,127,127,.18);border-radius:20px;padding:12px;backdrop-filter:blur(20px)}
</style>""",unsafe_allow_html=True)
