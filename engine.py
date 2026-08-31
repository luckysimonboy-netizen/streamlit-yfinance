import os,numpy as np,pandas as pd
import yfinance as yf
from data.market import market_snapshot,macro_snapshot,news_snapshot
from quant.engine import quant_snapshot
from valuation.engine import valuation_snapshot
from ai.orchestrator import AIOrchestrator
def secret(n,d=""):
    try:
        import streamlit as st
        v=st.secrets.get(n)
        if v is not None:return str(v)
    except Exception:pass
    return os.getenv(n,d)
class ResearchEngine:
    def __init__(self,ticker,period="2y"): self.ticker,self.period=ticker,period
    def build_pack(self):
        try:
            t=yf.Ticker(self.ticker); h=t.history(period=self.period,auto_adjust=False)
            i=t.get_info() or {}; news=t.get_news(count=30,tab="all") or []
        except Exception as e:return {"error":f"Data provider error: {e}"}
        if h.empty:return {"error":f"No market data returned for {self.ticker}"}
        q=quant_snapshot(h); m=market_snapshot(h,i); f=self._fundamental(i)
        v=valuation_snapshot(i,q["price"]); moat=self._moat(i); risk=self._risk(i,q)
        macro=macro_snapshot(); nr,ns=news_snapshot(news)
        score=float(np.clip(q["score"]*.25+f["score"]*.30+v["score"]*.15+moat*.10+ns*.08+(100-risk["score"])*.12,0,100))
        chart=pd.DataFrame({"Close":h["Close"],"MA20":h["Close"].rolling(20).mean(),"MA50":h["Close"].rolling(50).mean(),"MA200":h["Close"].rolling(200).mean()})
        warnings=[] if len(h)>=200 else ["Fewer than 200 sessions; long-term indicators are incomplete."]
        return {"error":None,"ticker":self.ticker,"company":i.get("longName") or i.get("shortName") or self.ticker,
        "summary":i.get("longBusinessSummary",""),"market":m,"macro":macro,"quant":q,"fundamental":f,"valuation":v,
        "moat":moat,"risk":risk,"news":nr,"news_score":ns,"score":score,"chart":chart,"quant_table":q["table"],
        "fundamental_table":self._fundamental_table(f),"providers":self._providers(),"warnings":warnings}
    def _providers(self):return {k:bool(secret(k+"_API_KEY")) for k in ["GEMINI","OPENROUTER","GROQ"]}
    def _fundamental(self,i):
        def x(k):
            try:
                z=float(i.get(k));return z if np.isfinite(z) else np.nan
            except Exception:return np.nan
        rg,eg,gm,om,pm,roe,de,cr,fcf=[x(k) for k in ["revenueGrowth","earningsGrowth","grossMargins","operatingMargins","profitMargins","returnOnEquity","debtToEquity","currentRatio","freeCashflow"]]
        growth=float(np.clip(50+(0 if np.isnan(rg) else rg*70)+(0 if np.isnan(eg) else eg*40),0,100))
        quality=50
        for val,hi,mid in [(gm,.5,.3),(om,.2,.1),(pm,.2,.1),(roe,.2,.1)]:
            if not np.isnan(val):quality+=10 if val>hi else 5 if val>mid else -7
        quality=float(np.clip(quality,0,100)); balance=60
        if not np.isnan(de):balance+=15 if de<50 else -20 if de>200 else 0
        if not np.isnan(cr):balance+=10 if cr>1.5 else -15 if cr<1 else 0
        balance=float(np.clip(balance,0,100)); cash=85 if not np.isnan(fcf) and fcf>0 else 30
        score=float(np.clip(growth*.30+quality*.35+balance*.20+cash*.15,0,100))
        return locals()
    def _fundamental_table(self,f):
        def pct(x):return "—" if np.isnan(x) else f"{x*100:.1f}%"
        return pd.DataFrame({"Metric":["Revenue Growth","EPS Growth","Gross Margin","Operating Margin","Profit Margin","ROE","Debt/Equity","Current Ratio","Free Cash Flow"],
        "Value":[pct(f["rg"]),pct(f["eg"]),pct(f["gm"]),pct(f["om"]),pct(f["pm"]),pct(f["roe"]),
        "—" if np.isnan(f["de"]) else f"{f['de']:.1f}","—" if np.isnan(f["cr"]) else f"{f['cr']:.2f}",
        "—" if np.isnan(f["fcf"]) else f"${f['fcf']:,.0f}"]})
    def _moat(self,i):
        vals=[]
        for k in ["grossMargins","operatingMargins","returnOnEquity","revenueGrowth"]:
            try:
                z=float(i.get(k))
                if np.isfinite(z):vals.append(np.clip(50+z*70,0,100))
            except Exception:pass
        return float(np.mean(vals)) if vals else 50.0
    def _risk(self,i,q):
        s=40
        try:
            beta=float(i.get("beta"));s+=25 if beta>1.7 else 15 if beta>1.3 else -10 if beta<.8 else 0
        except Exception:pass
        vol=q.get("volatility",np.nan)
        if np.isfinite(vol):s+=20 if vol>.6 else 10 if vol>.4 else -8 if vol<.2 else 0
        return {"score":float(np.clip(s,0,100))}
    def run_ai(self,pack):return AIOrchestrator().run(pack)
