import os
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

APP_VERSION = "Simon Stock V12.1 OTA"
DATA_TTL = 600
AI_TTL = 300

st.set_page_config(page_title=APP_VERSION, page_icon="✦", layout="wide", initial_sidebar_state="expanded")

# ============================================================
# Theme: system-following ColorOS x Liquid Glass
# CSS follows the device/browser preference via prefers-color-scheme.
# No manual light/dark toggle is required.
# ============================================================
st.markdown(r"""
<style>
:root{
  --bg:#f5f7fb; --bg2:#edf2f7; --text:#17202a; --muted:#667382;
  --glass:rgba(255,255,255,.68); --glass2:rgba(255,255,255,.50);
  --border:rgba(20,35,50,.10); --shadow:0 18px 55px rgba(30,50,70,.10);
  --accent:#1677ff; --good:#12945b; --bad:#d9445d; --warn:#a56a00;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#071018; --bg2:#0b1520; --text:#f5f7fb; --muted:#9ca8b6;
    --glass:rgba(255,255,255,.075); --glass2:rgba(255,255,255,.045);
    --border:rgba(255,255,255,.12); --shadow:0 18px 60px rgba(0,0,0,.28);
    --accent:#8fe7ff; --good:#64e6a2; --bad:#ff7e91; --warn:#ffd36e;
  }
}
.stApp{
  color:var(--text);
  background:
    radial-gradient(circle at 8% 0%, rgba(71,183,255,.16), transparent 30%),
    radial-gradient(circle at 92% 4%, rgba(146,104,255,.13), transparent 28%),
    linear-gradient(135deg,var(--bg) 0%,var(--bg2) 100%);
}
section[data-testid="stSidebar"]{
  background:color-mix(in srgb,var(--bg) 72%,transparent);
  backdrop-filter:blur(24px); border-right:1px solid var(--border);
}
.glass{
  background:linear-gradient(145deg,var(--glass),var(--glass2));
  border:1px solid var(--border); border-radius:24px; padding:20px;
  margin:8px 0 15px; box-shadow:var(--shadow); backdrop-filter:blur(22px);
}
.k{color:var(--muted);font-size:.76rem;letter-spacing:.09em;text-transform:uppercase}
.score{font-size:3.6rem;font-weight:850;letter-spacing:-.06em;line-height:1}
.muted,.small{color:var(--muted)}
.pill{display:inline-block;padding:5px 10px;border-radius:999px;background:var(--glass2);border:1px solid var(--border);margin:2px 4px 2px 0;font-size:.78rem}
.news-card{background:var(--glass2);border:1px solid var(--border);border-radius:18px;padding:15px 17px;margin:8px 0}
div[data-testid="stMetric"]{background:var(--glass2);border:1px solid var(--border);border-radius:18px;padding:10px}
button[data-testid="baseButton-secondary"],button[data-testid="baseButton-primary"]{border-radius:16px}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Utilities
# -----------------------------
def secret(key, default=""):
    try:
        value = st.secrets.get(key, None)
        if value is not None:
            return str(value)
    except Exception:
        pass
    return os.getenv(key, default)

def num(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan

def money(x):
    x = num(x)
    return "—" if np.isnan(x) else f"${x:,.2f}"

def integer_money(x):
    x = num(x)
    return "—" if np.isnan(x) else f"${x:,.0f}"

def pct(x):
    x = num(x)
    return "—" if np.isnan(x) else f"{x*100:.1f}%"

def ratio(x):
    x = num(x)
    return "—" if np.isnan(x) else f"{x:.2f}"

def clamp(x, lo=0, hi=100):
    x = num(x)
    return 50.0 if np.isnan(x) else float(np.clip(x, lo, hi))

def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def safe_json(obj):
    return json.dumps(obj, ensure_ascii=False, default=str)

def cache_id(prefix, payload):
    return hashlib.sha256((prefix + safe_json(payload)).encode()).hexdigest()[:24]

# -----------------------------
# Data layer
# -----------------------------
@st.cache_data(ttl=DATA_TTL, show_spinner=False)
def get_history(ticker, period="2y", interval="1d"):
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=DATA_TTL*2, show_spinner=False)
def get_info(ticker):
    try:
        return yf.Ticker(ticker).get_info() or {}
    except Exception:
        return {}

@st.cache_data(ttl=DATA_TTL, show_spinner=False)
def get_news(ticker):
    try:
        return yf.Ticker(ticker).get_news(count=50, tab="all") or []
    except Exception:
        return []

@st.cache_data(ttl=DATA_TTL, show_spinner=False)
def get_calendar(ticker):
    try:
        cal = yf.Ticker(ticker).calendar
        return cal if isinstance(cal, (pd.DataFrame, dict)) else {}
    except Exception:
        return {}

@st.cache_data(ttl=DATA_TTL, show_spinner=False)
def get_market_snapshot():
    symbols = {"S&P 500":"^GSPC","Nasdaq":"^IXIC","Dow":"^DJI","Russell 2000":"^RUT","VIX":"^VIX","US 10Y":"^TNX"}
    out = {}
    for name, ticker in symbols.items():
        try:
            df = get_history(ticker, "10d")
            if df.empty: continue
            c = num(df["Close"].iloc[-1]); p = num(df["Close"].iloc[-2]) if len(df)>1 else c
            out[name] = {"price":c,"change":(c/p-1) if p else np.nan}
        except Exception:
            continue
    return out

def validate(df, info):
    issues=[]
    if df.empty: issues.append("Price history unavailable.")
    if not df.empty and len(df)<80: issues.append("History is short; long-term indicators may be incomplete.")
    if not info: issues.append("Fundamental metadata unavailable.")
    if not df.empty and num(df["Close"].iloc[-1]) <= 0: issues.append("Latest price is invalid.")
    return issues

# -----------------------------
# Quant engine
# -----------------------------
def quant_engine(df):
    d=df.copy()
    c=d["Close"].astype(float); h=d["High"].astype(float); l=d["Low"].astype(float)
    for w in (20,50,100,200): d[f"MA{w}"]=c.rolling(w).mean()
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean()
    d["MACD"]=e12-e26; d["MACD_SIGNAL"]=d["MACD"].ewm(span=9,adjust=False).mean()
    delta=c.diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=-delta.clip(upper=0).rolling(14).mean()
    rs=gain/loss.replace(0,np.nan); d["RSI"]=100-(100/(1+rs))
    mid=c.rolling(20).mean(); sd=c.rolling(20).std(); d["BB_UPPER"]=mid+2*sd; d["BB_LOWER"]=mid-2*sd
    prev=c.shift(1); tr=pd.concat([(h-l),(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    d["ATR14"]=tr.rolling(14).mean(); d["VOL20"]=c.pct_change().rolling(20).std()*np.sqrt(252)
    d["M1"]=c/c.shift(21)-1; d["M3"]=c/c.shift(63)-1; d["M6"]=c/c.shift(126)-1; d["DD"]=c/c.cummax()-1
    x=d.iloc[-1]; score=50
    for w,p,n in ((20,5,-5),(50,8,-7),(200,12,-12)):
        ma=num(x.get(f"MA{w}"))
        if not np.isnan(ma): score += p if x["Close"]>ma else n
    rsi=num(x.get("RSI")); score += 8 if 42<=rsi<=65 else 3 if rsi<30 else -10 if rsi>75 else 0
    score += 7 if num(x.get("MACD"))>num(x.get("MACD_SIGNAL")) else -6
    m3=num(x.get("M3")); score += 8 if m3>.15 else 4 if m3>0 else -8 if m3<-.15 else -3
    return d, {"price":num(x["Close"]),"ma20":num(x.get("MA20")),"ma50":num(x.get("MA50")),"ma100":num(x.get("MA100")),"ma200":num(x.get("MA200")),"rsi":rsi,"macd":num(x.get("MACD")),"signal":num(x.get("MACD_SIGNAL")),"atr":num(x.get("ATR14")),"vol":num(x.get("VOL20")),"m1":num(x.get("M1")),"m3":num(x.get("M3")),"m6":num(x.get("M6")),"dd":num(x.get("DD")),"score":clamp(score)}

# -----------------------------
# Fundamentals / company profile
# -----------------------------
def fundamentals(i):
    v=lambda k:num(i.get(k))
    keys=["revenueGrowth","earningsGrowth","grossMargins","operatingMargins","profitMargins","returnOnEquity","returnOnAssets","debtToEquity","currentRatio","trailingPE","forwardPE","pegRatio","priceToSalesTrailing12Months","priceToBook","enterpriseToEbitda"]
    rg,eg,gm,om,pm,roe,roa,de,cr,pe,fpe,peg,ps,pb,ev=[v(k) for k in keys]
    growth=clamp(50+(np.clip(rg*80,-25,25) if not np.isnan(rg) else 0)+(np.clip(eg*50,-20,20) if not np.isnan(eg) else 0))
    quality=50
    quality += 12 if not np.isnan(gm) and gm>.5 else 7 if not np.isnan(gm) and gm>.3 else -5 if not np.isnan(gm) else 0
    quality += 15 if not np.isnan(om) and om>.2 else 8 if not np.isnan(om) and om>.1 else -8 if not np.isnan(om) else 0
    quality += 12 if not np.isnan(pm) and pm>.2 else 6 if not np.isnan(pm) and pm>.1 else -6 if not np.isnan(pm) else 0
    quality += 12 if not np.isnan(roe) and roe>.2 else 6 if not np.isnan(roe) and roe>.1 else -6 if not np.isnan(roe) else 0
    balance=60
    balance += 15 if not np.isnan(de) and de<50 else -20 if not np.isnan(de) and de>200 else 0
    balance += 10 if not np.isnan(cr) and cr>=1.5 else -15 if not np.isnan(cr) and cr<1 else 0
    fcf=v("freeCashflow"); cash=85 if not np.isnan(fcf) and fcf>0 else 30
    val=55
    if not np.isnan(peg): val += 20 if peg<1 else 10 if peg<1.5 else -20 if peg>2.5 else 0
    elif not np.isnan(pe): val += 15 if pe<18 else -20 if pe>40 else -10 if pe>30 else 0
    if not np.isnan(pe) and not np.isnan(fpe): val += 8 if fpe<pe else -8
    total=.25*growth+.25*clamp(quality)+.15*clamp(balance)+.15*cash+.20*clamp(val)
    return {"score":clamp(total),"growth":growth,"quality":clamp(quality),"balance":clamp(balance),"cash":cash,"valuation":clamp(val),"revenue_growth":rg,"earnings_growth":eg,"gross_margin":gm,"operating_margin":om,"profit_margin":pm,"roe":roe,"roa":roa,"debt_equity":de,"current_ratio":cr,"pe":pe,"forward_pe":fpe,"peg":peg,"ps":ps,"pb":pb,"ev_ebitda":ev,"fcf":fcf}

def company_metrics(i, q):
    return {
        "Market Cap":integer_money(i.get("marketCap")), "Enterprise Value":integer_money(i.get("enterpriseValue")),
        "52W High":money(i.get("fiftyTwoWeekHigh")), "52W Low":money(i.get("fiftyTwoWeekLow")),
        "Beta":ratio(i.get("beta")), "Average Volume":f"{num(i.get('averageVolume')):,.0f}" if not np.isnan(num(i.get('averageVolume'))) else "—",
        "Dividend Yield":pct(i.get("dividendYield")), "Dividend Rate":money(i.get("dividendRate")),
        "Target Mean":money(i.get("targetMeanPrice")), "Target Low":money(i.get("targetLowPrice")), "Target High":money(i.get("targetHighPrice")),
        "Analyst Rating":str(i.get("recommendationKey") or "—"), "Analysts":str(i.get("numberOfAnalystOpinions") or "—"),
        "Shares Outstanding":f"{num(i.get('sharesOutstanding')):,.0f}" if not np.isnan(num(i.get('sharesOutstanding'))) else "—",
        "Institutional":pct(i.get("heldPercentInstitutions")), "Insider":pct(i.get("heldPercentInsiders")),
        "Short % Float":pct(i.get("shortPercentOfFloat")), "Earnings Growth":pct(i.get("earningsGrowth")),
    }

# -----------------------------
# News engine: robust parsing
# -----------------------------
POS=["beat","beats","upgrade","upgraded","bullish","record","surge","strong","profit","approval","buyback","raise","raised","outperform","partnership","growth"]
NEG=["miss","misses","downgrade","downgraded","bearish","lawsuit","investigation","decline","weak","loss","cut","warning","recall","layoff","fraud","delay","risk"]

def parse_news(items):
    rows=[]
    for item in items:
        c=item.get("content",item) if isinstance(item,dict) else {}
        title=str(c.get("title") or item.get("title") or "").strip()
        if not title: continue
        low=title.lower(); s=sum(w in low for w in POS)-sum(w in low for w in NEG)
        sentiment="Bullish" if s>0 else "Bearish" if s<0 else "Neutral"
        provider=c.get("provider"); pub=provider.get("displayName") if isinstance(provider,dict) else str(item.get("publisher") or "Yahoo Finance")
        url=""
        for k in ("clickThroughUrl","canonicalUrl"):
            u=c.get(k)
            if isinstance(u,dict) and u.get("url"): url=u["url"]; break
        pubdate=c.get("pubDate") or c.get("displayTime") or item.get("providerPublishTime")
        rows.append({"title":title,"sentiment":sentiment,"score":s,"publisher":pub,"url":url,"date":str(pubdate or "")})
    avg=np.mean([x["score"] for x in rows]) if rows else 0
    return rows,clamp(50+avg*15)

# -----------------------------
# AI Gateway: resilient, cached, provider-aware
# -----------------------------
def provider_status():
    return {"Gemini":bool(secret("GEMINI_API_KEY")),"OpenRouter":bool(secret("OPENROUTER_API_KEY")),"Groq":bool(secret("GROQ_API_KEY"))}

def ai_order():
    forced=secret("AI_PROVIDER","auto").lower()
    names=["Gemini","OpenRouter","Groq"]
    if forced in {x.lower() for x in names}:
        first=forced.title(); return [first]+[x for x in names if x!=first]
    return names

def call_gemini(prompt):
    from google import genai
    client=genai.Client(api_key=secret("GEMINI_API_KEY"))
    model=secret("GEMINI_MODEL","gemini-3.7-flash")
    interaction=client.interactions.create(model=model,input=prompt)
    return getattr(interaction,"output_text",None) or str(interaction)

def call_openrouter(prompt):
    import requests
    key=secret("OPENROUTER_API_KEY"); model=secret("OPENROUTER_MODEL","openrouter/free")
    headers={"Authorization":f"Bearer {key}","Content-Type":"application/json","HTTP-Referer":secret("APP_URL","https://streamlit.io"),"X-Title":"Simon Stock"}
    body={"model":model,"messages":[{"role":"system","content":"You are Simon Stock AI. Never invent financial data. Separate facts, assumptions, inferences and risks. Never guarantee returns."},{"role":"user","content":prompt}]}
    r=requests.post("https://openrouter.ai/api/v1/chat/completions",headers=headers,json=body,timeout=60); r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def call_groq(prompt):
    import requests
    key=secret("GROQ_API_KEY"); model=secret("GROQ_MODEL","openai/gpt-oss-120b")
    headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"}
    body={"model":model,"messages":[{"role":"system","content":"You are Simon Stock AI. Never invent financial data. Separate facts, assumptions, inferences and risks. Never guarantee returns."},{"role":"user","content":prompt}]}
    r=requests.post("https://api.groq.com/openai/v1/chat/completions",headers=headers,json=body,timeout=60); r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

@st.cache_data(ttl=AI_TTL,show_spinner=False)
def ai_request(prompt_hash,prompt):
    status=provider_status(); errors=[]
    for p in ai_order():
        if not status.get(p): continue
        try:
            text=call_gemini(prompt) if p=="Gemini" else call_openrouter(prompt) if p=="OpenRouter" else call_groq(prompt)
            return {"ok":True,"provider":p,"text":text,"error":""}
        except Exception as e:
            errors.append(f"{p}: {type(e).__name__}: {e}")
    return {"ok":False,"provider":None,"text":"No working AI provider.","error":" | ".join(errors)}

def ask_ai(prompt):
    return ai_request(hashlib.sha256(prompt.encode()).hexdigest()[:24],prompt)

# -----------------------------
# Apex research workflow
# -----------------------------
def research_pack(ticker,i,q,f,moat,risk,news_score,news_rows,validation):
    return {"as_of":now_utc(),"ticker":ticker,"company":i.get("longName") or ticker,"sector":i.get("sector"),"industry":i.get("industry"),"price":q["price"],"quant":q,"fundamental":f,"moat":moat,"risk":risk,"news_score":news_score,"news":[{"title":x["title"],"sentiment":x["sentiment"],"publisher":x["publisher"]} for x in news_rows[:20]],"company_metrics":company_metrics(i,q),"business_summary":i.get("longBusinessSummary",""),"validation":validation}

def specialist_prompt(role,pack):
    return f"""You are the {role} specialist inside Simon Stock V12 AI Apex.\nUse ONLY the supplied research pack. Never invent missing numbers.\nAnalyze evidence, assumptions, risks, counter-evidence and what would falsify the thesis.\nDo not reveal private chain-of-thought; provide concise reasoning summaries.\n\nRESEARCH PACK:\n{safe_json(pack)}"""

def run_apex(pack):
    roles=["Fundamental Analyst","Quant Analyst","Valuation Analyst","Macro Analyst","News Analyst","Risk Officer"]
    reports={}
    for role in roles:
        reports[role]=ask_ai(specialist_prompt(role,pack))
    joined="\n\n".join(f"### {k}\n{v.get('text','')}" for k,v in reports.items())
    bull=ask_ai(f"""You are the Bull Analyst. Build the strongest evidence-based bullish thesis from these specialist reports. Then identify the strongest evidence that could break it. Do not invent numbers.\n{joined}""")
    bear=ask_ai(f"""You are the Bear Analyst. Attack the bullish thesis using the supplied evidence. Identify valuation, execution, macro and business risks. Then state what would invalidate your bear case. Do not invent numbers.\n{joined}\nBULL:\n{bull.get('text','')}""")
    judge=ask_ai(f"""You are the Chief Analyst / AI Judge. Adjudicate the specialists plus Bull and Bear. Do not average opinions blindly. Weight hard data over speculation.\nReturn exactly these headings:\nFINAL VERDICT\nSCORE\nCONFIDENCE\nTOP REASONS\nTOP RISKS\nWHAT CHANGES MY MIND\nDATA QUALITY WARNING\nNo guaranteed returns and no fabricated price target.\n\nSPECIALISTS:\n{joined}\n\nBULL:\n{bull.get('text','')}\n\nBEAR:\n{bear.get('text','')}""")
    return {"reports":reports,"bull":bull,"bear":bear,"judge":judge}

# -----------------------------
# State isolation per ticker
# -----------------------------
if "apex_by_ticker" not in st.session_state: st.session_state.apex_by_ticker={}
if "thesis_by_ticker" not in st.session_state: st.session_state.thesis_by_ticker={}

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## ✦ Simon Stock")
    st.caption(APP_VERSION)
    ticker=st.text_input("Ticker","AAPL").strip().upper()
    period=st.selectbox("Price history",["6mo","1y","2y","5y"],index=2)
    st.divider()
    st.markdown("### AI Gateway")
    status=provider_status()
    for p,ok in status.items(): st.write(("🟢 " if ok else "⚪ ")+p)
    if any(status.values()): st.success("AI ready")
    else: st.info("AI keys not configured")
    run_ai=st.button("✦ Run Apex AI",type="primary",use_container_width=True)
    refresh=st.button("↻ Refresh market data",use_container_width=True)
    if refresh:
        st.cache_data.clear(); st.rerun()
    st.divider()
    st.caption("Theme follows your device/browser system preference. No separate theme switch is required.")

# -----------------------------
# Load + calculate
# -----------------------------
df=get_history(ticker,period)
info=get_info(ticker)
if df.empty:
    st.error(f"无法取得 {ticker} 的行情数据。请检查 ticker 或点击左侧刷新。")
    st.stop()
validation=validate(df,info)
qdf,q=quant_engine(df)
f=fundamentals(info)
moat=clamp(50+(15 if f["quality"]>80 else 8 if f["quality"]>65 else -8)+(12 if f["growth"]>75 else 5 if f["growth"]>60 else -8))
beta=num(info.get("beta")); risk=clamp(40+(25 if beta>1.7 else 15 if beta>1.3 else -10 if beta<.8 else 0)+(20 if q["vol"]>.6 else 10 if q["vol"]>.4 else -8 if q["vol"]<.2 else 0))
news_rows,news_score=parse_news(get_news(ticker))
hype=clamp(max(q["rsi"]-65,0)*2+max(q["m1"]-.15,0)*100+max(news_score-75,0)*.5)
score=clamp(q["score"]*.25+f["score"]*.30+f["valuation"]*.10+moat*.12+news_score*.08+(100-risk)*.10+(100-hype)*.05)
verdict="STRONG BUY" if score>=90 else "BUY" if score>=80 else "BUY ON DIPS" if score>=70 else "HOLD" if score>=60 else "WAIT" if score>=50 else "REDUCE / AVOID"

# -----------------------------
# Header / market strip
# -----------------------------
st.markdown(f'<div class="glass"><div class="k">AI-NATIVE EQUITY RESEARCH · {APP_VERSION}</div><h1>{ticker} · {info.get("longName",ticker)}</h1><div class="muted">{info.get("sector","Unknown")} · {info.get("industry","Unknown")} · Data {now_utc()}</div></div>',unsafe_allow_html=True)
market=get_market_snapshot()
if market:
    cols=st.columns(len(market))
    for col,(name,v) in zip(cols,market.items()):
        col.metric(name,money(v["price"]),f"{v['change']*100:+.2f}%")

m1,m2,m3,m4,m5=st.columns(5)
m1.metric("Price",money(q["price"])); m2.metric("Simon Score",f"{score:.0f}/100"); m3.metric("Verdict",verdict); m4.metric("Risk",f"{risk:.0f}/100"); m5.metric("News",f"{news_score:.0f}/100")

# -----------------------------
# Main tabs
# -----------------------------
tabs=st.tabs(["✦ Apex AI","📊 Quant","🏢 Fundamentals","🏷️ Company","📰 Daily News","🧠 Thesis","⚙️ System"])

with tabs[0]:
    a,b,c=st.columns(3)
    with a: st.markdown(f'<div class="glass"><div class="k">Simon Score</div><div class="score">{score:.0f}</div><b>{verdict}</b><div class="muted">Quant {q["score"]:.0f} · Fundamental {f["score"]:.0f}</div></div>',unsafe_allow_html=True)
    with b: st.markdown(f'<div class="glass"><div class="k">Research Stack</div><span class="pill">Fundamental {f["score"]:.0f}</span><span class="pill">Moat {moat:.0f}</span><span class="pill">Valuation {f["valuation"]:.0f}</span><span class="pill">Risk {risk:.0f}</span><span class="pill">News {news_score:.0f}</span></div>',unsafe_allow_html=True)
    with c: st.markdown(f'<div class="glass"><div class="k">Market Context</div><div>RSI <b>{q["rsi"]:.1f}</b></div><div>3M Momentum <b>{pct(q["m3"])}</b></div><div>Drawdown <b>{pct(q["dd"])}</b></div><div>Volatility <b>{pct(q["vol"])}</b></div></div>',unsafe_allow_html=True)

    if run_ai:
        if not any(status.values()):
            st.warning("没有配置 AI Key。Quant / Fundamental / News 仍可正常使用；配置 Secrets 后即可运行 Apex AI。")
        else:
            pack=research_pack(ticker,info,q,f,moat,risk,news_score,news_rows,validation)
            with st.spinner("① Specialist analysts..."):
                apex=run_apex(pack)
            st.session_state.apex_by_ticker[ticker]=apex
            st.session_state.thesis_by_ticker[ticker]={"timestamp":now_utc(),"score":score,"verdict":verdict,"price":q["price"],"judge":apex["judge"].get("text","")}
    apex=st.session_state.apex_by_ticker.get(ticker)
    if apex:
        judge=apex["judge"]
        if judge.get("ok"): st.success(f"AI Judge · {judge.get('provider')}")
        else: st.error(f"AI Judge unavailable: {judge.get('error','unknown error')}")
        st.markdown("### ⚖️ Chief Analyst"); st.markdown(judge.get("text",""))
        st.markdown("### 🐂 Bull Case"); st.markdown(apex["bull"].get("text",""))
        st.markdown("### 🐻 Bear Case"); st.markdown(apex["bear"].get("text",""))
        with st.expander("Specialist reports"):
            for role,res in apex["reports"].items():
                st.markdown(f"#### {role} · {res.get('provider') or 'unavailable'}")
                st.markdown(res.get("text",res.get("error","")))
    else:
        st.info("点击左侧 **Run Apex AI**，启动 Fundamental / Quant / Valuation / Macro / News / Risk → Bull/Bear → AI Judge。")

with tabs[1]:
    st.subheader("Quant Engine")
    chart_cols=[x for x in ["Close","MA20","MA50","MA100","MA200"] if x in qdf.columns]
    st.line_chart(qdf[chart_cols].dropna(how="all"))
    cols=st.columns(7)
    for col,label,value in zip(cols,["RSI","MA20","MA50","MA200","1M","3M","6M"],[f"{q['rsi']:.1f}",money(q['ma20']),money(q['ma50']),money(q['ma200']),pct(q['m1']),pct(q['m3']),pct(q['m6'])]): col.metric(label,value)
    st.dataframe(qdf.tail(60)[[x for x in ["Close","MA20","MA50","MA100","MA200","RSI","MACD","MACD_SIGNAL","ATR14","VOL20","M1","M3","M6","DD"] if x in qdf.columns]],use_container_width=True)

with tabs[2]:
    st.subheader("Fundamental & Valuation")
    rows=[
        ("Revenue Growth",pct(f["revenue_growth"])),("Earnings Growth",pct(f["earnings_growth"])),("Gross Margin",pct(f["gross_margin"])),
        ("Operating Margin",pct(f["operating_margin"])),("Profit Margin",pct(f["profit_margin"])),("ROE",pct(f["roe"])),("ROA",pct(f["roa"])),
        ("Debt / Equity",ratio(f["debt_equity"])),("Current Ratio",ratio(f["current_ratio"])),("P/E",ratio(f["pe"])),("Forward P/E",ratio(f["forward_pe"])),
        ("PEG",ratio(f["peg"])),("P/S",ratio(f["ps"])),("P/B",ratio(f["pb"])),("EV / EBITDA",ratio(f["ev_ebitda"])),("Free Cash Flow",integer_money(f["fcf"]))]
    st.dataframe(pd.DataFrame(rows,columns=["Metric","Value"]),use_container_width=True,hide_index=True)
    st.markdown(f'<div class="glass"><div class="k">First Principles · Business</div><p>{info.get("longBusinessSummary","Business summary unavailable.")}</p></div>',unsafe_allow_html=True)

with tabs[3]:
    st.subheader("Company Snapshot")
    cm=company_metrics(info,q)
    cdf=pd.DataFrame(list(cm.items()),columns=["Metric","Value"])
    st.dataframe(cdf,use_container_width=True,hide_index=True)
    cc1,cc2,cc3=st.columns(3)
    cc1.metric("52W Position",f"{((q['price']-num(info.get('fiftyTwoWeekLow')))/(num(info.get('fiftyTwoWeekHigh'))-num(info.get('fiftyTwoWeekLow')))*100):.0f}%" if not np.isnan(num(info.get('fiftyTwoWeekLow'))) and not np.isnan(num(info.get('fiftyTwoWeekHigh'))) and num(info.get('fiftyTwoWeekHigh'))>num(info.get('fiftyTwoWeekLow')) else "—")
    cc2.metric("Analyst View",str(info.get("recommendationKey") or "—"))
    cc3.metric("Target Mean",money(info.get("targetMeanPrice")))

with tabs[4]:
    st.subheader("Daily Stock News")
    st.metric("News Sentiment",f"{news_score:.0f}/100")
    if news_rows:
        for x in news_rows[:25]:
            icon="🟢" if x["sentiment"]=="Bullish" else "🔴" if x["sentiment"]=="Bearish" else "🟡"
            source=x["publisher"] or "Yahoo Finance"
            link=f' · <a href="{x["url"]}" target="_blank">Open</a>' if x["url"] else ""
            st.markdown(f'<div class="news-card"><b>{icon} {x["title"]}</b><br><span class="small">{source}{link}</span></div>',unsafe_allow_html=True)
    else: st.info("No news returned by the data provider.")

with tabs[5]:
    st.subheader("Thesis Memory")
    thesis=st.session_state.thesis_by_ticker.get(ticker)
    if thesis:
        st.write(f"Last thesis · {thesis['timestamp']} · {money(thesis['price'])} · Score {thesis['score']:.0f} · {thesis['verdict']}")
        st.markdown(thesis["judge"])
    else:
        st.info("Run Apex AI once to create a local thesis snapshot for this ticker.")
    st.caption("V12.1 keeps session thesis memory isolated by ticker. A future OTA can add persistent storage and automatic thesis re-evaluation.")

with tabs[6]:
    st.subheader("System / Data Health")
    st.write("Version",APP_VERSION)
    st.write("Data timestamp",now_utc())
    st.write("AI providers",status)
    st.write("Data warnings",validation if validation else "PASS")
    st.write("Theme", "System preference (prefers-color-scheme)")
    st.write("AI cache",f"{AI_TTL}s")
    if any(status.values()): st.success("At least one AI provider is configured.")
    else: st.warning("No AI provider key configured. Add keys to Streamlit Secrets.")
    st.info("AI outputs are research assistance, not financial advice. Never commit API keys to GitHub.")

st.divider()
st.caption(f"{APP_VERSION} · Yahoo Finance data · AI research assistance, not financial advice")
