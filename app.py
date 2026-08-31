import streamlit as st
from core.engine import ResearchEngine
from core.theme import inject_theme
st.set_page_config(page_title="Simon Stock V13",page_icon="✦",layout="wide")
inject_theme()
st.title("✦ Simon Stock V13")
st.caption("AI-native US equity research foundation")
with st.sidebar:
    ticker=st.text_input("Ticker","AAPL").strip().upper()
    period=st.selectbox("History",["6mo","1y","2y","5y"],index=2)
    st.select_slider("Information density",["Comfort","Balanced","Pro"],value="Balanced")
    run=st.button("Run Full AI Research",type="primary",use_container_width=True)
engine=ResearchEngine(ticker,period)
with st.spinner("Loading market, fundamentals and news…"):
    pack=engine.build_pack()
if pack["error"]:
    st.error(pack["error"]); st.stop()
m,f,q,v,r=pack["market"],pack["fundamental"],pack["quant"],pack["valuation"],pack["risk"]
a,b,c,d,e=st.columns(5)
a.metric("Price",f"${m['price']:,.2f}"); b.metric("Simon Score",f"{pack['score']:.0f}/100")
c.metric("Technical",f"{q['score']:.0f}"); d.metric("Fundamental",f"{f['score']:.0f}"); e.metric("Risk",f"{r['score']:.0f}")
tabs=st.tabs(["AI Committee","Overview","Quant","Fundamentals","Valuation","Daily News","System"])
with tabs[0]:
    if run:
        with st.spinner("Specialists → Bull/Bear → Chief Judge…"):
            st.session_state["ai"]=engine.run_ai(pack)
    ai=st.session_state.get("ai")
    if ai:
        st.markdown(f"### {ai['verdict']} · {ai['score']}/100")
        st.caption(f"Confidence {ai['confidence']}/100 · {ai['provider']}")
        st.markdown(ai["judge"])
        with st.expander("Bull / Bear debate"): st.markdown(ai["debate"])
        with st.expander("Specialist reports"):
            for name,text in ai["specialists"].items():
                st.markdown(f"#### {name}"); st.markdown(text)
    else: st.info("Run Full AI Research to activate the committee.")
with tabs[1]:
    st.subheader(pack["company"]); st.write(pack["summary"] or "Business summary unavailable.")
    a,b,c,d=st.columns(4)
    a.metric("Market Cap",m["market_cap"]); b.metric("52W High",m["52w_high"])
    c.metric("52W Low",m["52w_low"]); d.metric("Beta",m["beta"])
    st.json(pack["macro"])
with tabs[2]:
    st.line_chart(pack["chart"]); st.dataframe(pack["quant_table"],use_container_width=True,hide_index=True)
with tabs[3]:
    st.dataframe(pack["fundamental_table"],use_container_width=True,hide_index=True)
    a,b,c,d=st.columns(4)
    a.metric("Quality",f"{f['quality']:.0f}"); b.metric("Growth",f"{f['growth']:.0f}")
    c.metric("Balance",f"{f['balance']:.0f}"); d.metric("Moat",f"{pack['moat']:.0f}")
with tabs[4]:
    st.metric("Valuation Score",f"{v['score']:.0f}/100")
    st.write(f"Trailing P/E: {v['pe']} · Forward P/E: {v['forward_pe']} · PEG: {v['peg']} · FCF Yield: {v['fcf_yield']}")
with tabs[5]:
    st.metric("News Sentiment",f"{pack['news_score']:.0f}/100")
    for x in pack["news"]:
        st.markdown(f"**{x['sentiment']} · {x['title']}**"); st.caption(x["publisher"])
with tabs[6]:
    st.write("Pipeline: Data → Quant → Fundamental → Valuation → Risk → AI Committee")
    st.write("AI providers:",pack["providers"]); st.write("Warnings:",pack["warnings"])
    st.info("Research assistance only; not financial advice.")
