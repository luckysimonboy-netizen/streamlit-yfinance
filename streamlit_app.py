import os, json, hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

VERSION="Simon Stock V12 AI Apex"
TTL=900

st.set_page_config(page_title=VERSION,page_icon="✦",layout="wide")
st.markdown("""<style>
.stApp{background:radial-gradient(circle at 10% 0%,#14334a 0,transparent 30%),radial-gradient(circle at 90% 0%,#30224b 0,transparent 28%),#071018;color:#f5f7fb}
section[data-testid="stSidebar"]{background:rgba(5,10,16,.78);backdrop-filter:blur(24px)}
.glass{background:linear-gradient(145deg,rgba(255,255,255,.11),rgba(255,255,255,.035));border:1px solid rgba(255,255,255,.12);border-radius:24px;padding:20px;margin:8px 0 14px;box-shadow:0 18px 60px #0004;backdrop-filter:blur(22px)}
.k{color:#9ca8b6;font-size:.78rem;letter-spacing:.08em;text-transform:uppercase}.score{font-size:3.5rem;font-weight:800;letter-spacing:-.06em}.muted{color:#9ca8b6}
</style>""",unsafe_allow_html=True)

def sec(k,d=""):
    try:
        v=st.secrets.get(k,None)
        if v is not None:return str(v)
    except:pass
    return os.getenv(k,d)
def num(x):
    try:
        x=float(x); return x if np.isfinite(x) else np.nan
    except:return np.nan
def money(x):
    x=num(x); return "—" if np.isnan(x) else f"${x:,.2f}"
def percent(x):
    x=num(x); return "—" if np.isnan(x) else f"{x*100:.1f}%"
def clamp(x): return float(np.clip(num(x) if not np.isnan(num(x)) else 50,0,100))

@st.cache_data(ttl=TTL,show_spinner=False)
def hist(ticker,period):
    try:return yf.Ticker(ticker).history(period=period,auto_adjust=False)
    except:return pd.DataFrame()
@st.cache_data(ttl=TTL*2,show_spinner=False)
def info(ticker):
    try:return yf.Ticker(ticker).get_info() or {}
    except:return {}
@st.cache_data(ttl=TTL,show_spinner=False)
def news(ticker):
    try:return yf.Ticker(ticker).get_news(count=30,tab="all") or []
    except:return []

def quant(d):
    d=d.copy()
    c=d["Close"].astype(float); h=d["High"].astype(float); l=d["Low"].astype(float)
    for w in [20,50,200]: d[f"MA{w}"]=c.rolling(w).mean()
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean()
    d["MACD"]=e12-e26; d["SIGNAL"]=d["MACD"].ewm(span=9,adjust=False).mean()
    delta=c.diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=-delta.clip(upper=0).rolling(14).mean()
    d["RSI"]=100-100/(1+(gain/loss.replace(0,np.nan)))
    prev=c.shift(); tr=pd.concat([h-l,(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    d["ATR"]=tr.rolling(14).mean(); d["VOL"]=c.pct_change().rolling(20).std()*np.sqrt(252)
    d["M1"]=c/c.shift(21)-1; d["M3"]=c/c.shift(63)-1; d["M6"]=c/c.shift(126)-1
    d["DD"]=c/c.cummax()-1
    x=d.iloc[-1]; s=50
    for w,p,n in [(20,5,-5),(50,8,-7),(200,12,-12)]:
        ma=num(x.get(f"MA{w}")); s+=(p if x["Close"]>ma else n) if not np.isnan(ma) else 0
    r=num(x.RSI); s+=8 if 42<=r<=65 else 3 if r<30 else -10 if r>75 else 0
    s+=7 if num(x.MACD)>num(x.SIGNAL) else -6
    m=num(x.M3); s+=8 if m>.15 else 4 if m>0 else -8 if m<-.15 else -3
    return d,{"price":num(x.Close),"rsi":r,"ma20":num(x.MA20),"ma50":num(x.MA50),"ma200":num(x.MA200),
              "macd":num(x.MACD),"signal":num(x.SIGNAL),"vol":num(x.VOL),"m1":num(x.M1),"m3":num(x.M3),
              "m6":num(x.M6),"dd":num(x.DD),"score":clamp(s)}

def fundamentals(i):
    def v(k):return num(i.get(k))
    rg,eg,gm,om,pm,roe,de,cr,pe,fpe,peg=[v(k) for k in
        ["revenueGrowth","earningsGrowth","grossMargins","operatingMargins","profitMargins","returnOnEquity","debtToEquity","currentRatio","trailingPE","forwardPE","pegRatio"]]
    growth=clamp(50+(np.clip(rg*80,-25,25) if not np.isnan(rg) else 0)+(np.clip(eg*50,-20,20) if not np.isnan(eg) else 0))
    quality=50+(12 if not np.isnan(gm) and gm>.5 else 7 if not np.isnan(gm) and gm>.3 else -5 if not np.isnan(gm) else 0)
    quality+=(15 if not np.isnan(om) and om>.2 else 8 if not np.isnan(om) and om>.1 else -8 if not np.isnan(om) else 0)
    quality+=(12 if not np.isnan(pm) and pm>.2 else 6 if not np.isnan(pm) and pm>.1 else -6 if not np.isnan(pm) else 0)
    quality+=(12 if not np.isnan(roe) and roe>.2 else 6 if not np.isnan(roe) and roe>.1 else -6 if not np.isnan(roe) else 0)
    balance=60+(15 if not np.isnan(de) and de<50 else -20 if not np.isnan(de) and de>200 else 0)
    balance+=(10 if not np.isnan(cr) and cr>=1.5 else -15 if not np.isnan(cr) and cr<1 else 0)
    cash=85 if num(i.get("freeCashflow"))>0 else 30
    val=55
    if not np.isnan(peg):val+=20 if peg<1 else 10 if peg<1.5 else -20 if peg>2.5 else 0
    elif not np.isnan(pe):val+=15 if pe<18 else -20 if pe>40 else -10 if pe>30 else 0
    if not np.isnan(pe) and not np.isnan(fpe):val+=8 if fpe<pe else -8
    total=.25*growth+.25*clamp(quality)+.15*clamp(balance)+.15*cash+.20*clamp(val)
    return {"score":clamp(total),"growth":growth,"quality":clamp(quality),"balance":clamp(balance),"cash":cash,"valuation":clamp(val),
            "revenue_growth":rg,"earnings_growth":eg,"gross_margin":gm,"operating_margin":om,"profit_margin":pm,"roe":roe,
            "debt_equity":de,"current_ratio":cr,"pe":pe,"forward_pe":fpe,"peg":peg,"fcf":num(i.get("freeCashflow"))}

POS={"beat","upgrade","bullish","record","surge","strong","profit","approval","buyback","raised","outperform","partnership"}
NEG={"miss","downgrade","bearish","lawsuit","investigation","decline","weak","loss","cut","warning","recall","layoff","fraud","delay"}
def news_engine(items):
    rows=[]
    for item in items:
        c=item.get("content",item); title=str(c.get("title") or item.get("title") or "").strip()
        if not title:continue
        low=title.lower(); p=sum(w in low for w in POS); n=sum(w in low for w in NEG)
        s=p-n; sentiment="Bullish" if s>0 else "Bearish" if s<0 else "Neutral"
        provider=c.get("provider"); pub=provider.get("displayName") if isinstance(provider,dict) else "Yahoo Finance"
        url=""
        for k in ("clickThroughUrl","canonicalUrl"):
            u=c.get(k); 
            if isinstance(u,dict) and u.get("url"):url=u["url"];break
        rows.append({"title":title,"sentiment":sentiment,"publisher":pub,"url":url,"score":s})
    return rows,clamp(50+(np.mean([x["score"] for x in rows]) if rows else 0)*15)

def providers():
    return {"Gemini":bool(sec("GEMINI_API_KEY")),"OpenRouter":bool(sec("OPENROUTER_API_KEY")),"Groq":bool(sec("GROQ_API_KEY"))}

def ai(prompt):
    stt=providers()
    order=["Gemini","OpenRouter","Groq"]
    forced=sec("AI_PROVIDER","auto").lower()
    if forced in ["gemini","openrouter","groq"]:
        order=[forced.title()]+[x for x in order if x.lower()!=forced]
    errs=[]
    for p in order:
        if not stt[p]:continue
        try:
            if p=="Gemini":
                from google import genai
                client=genai.Client(api_key=sec("GEMINI_API_KEY"))
                r=client.interactions.create(model=sec("GEMINI_MODEL","gemini-3.7-flash"),input=prompt)
                return {"ok":True,"provider":p,"text":r.output_text}
            import requests
            key=sec("OPENROUTER_API_KEY") if p=="OpenRouter" else sec("GROQ_API_KEY")
            base="https://openrouter.ai/api/v1" if p=="OpenRouter" else "https://api.groq.com/openai/v1"
            model=sec("OPENROUTER_MODEL","openrouter/free") if p=="OpenRouter" else sec("GROQ_MODEL","openai/gpt-oss-120b")
            h={"Authorization":f"Bearer {key}","Content-Type":"application/json"}
            if p=="OpenRouter":h["HTTP-Referer"]=sec("APP_URL","https://streamlit.io")
            body={"model":model,"messages":[{"role":"system","content":"You are Simon Stock AI. Never invent data. Separate facts, assumptions and conclusions. Do not guarantee returns."},{"role":"user","content":prompt}]}
            r=requests.post(base+"/chat/completions",headers=h,json=body,timeout=60); r.raise_for_status()
            return {"ok":True,"provider":p,"text":r.json()["choices"][0]["message"]["content"]}
        except Exception as e:errs.append(f"{p}: {e}")
    return {"ok":False,"provider":None,"text":"No working AI provider. "+(" | ".join(errs) if errs else "Add an API key in Streamlit Secrets.")}

def prompt_for(ticker,pack):
    return f"""You are the {t} specialist inside Simon Stock V12 AI Apex.
Use ONLY the supplied research data. Never invent missing numbers.
Analyze first principles, evidence, assumptions, risks and what would falsify the thesis.
Do not reveal private chain-of-thought; give concise reasoning summaries.
DATA:
{json.dumps(pack,ensure_ascii=False,default=str,indent=2)}"""

with st.sidebar:
    st.markdown("## ✦ Simon Stock")
    st.caption(VERSION)
    ticker=st.text_input("Ticker","AAPL").strip().upper()
    period=st.selectbox("History",["6mo","1y","2y","5y"],2)
    st.divider()
    st.write("AI Gateway")
    for p,v in providers().items():st.write(("🟢 " if v else "⚪ ")+p)
    run=st.button("✦ Run Apex AI",type="primary",use_container_width=True)

d=hist(ticker,period); i=info(ticker)
if d.empty:
    st.error("无法取得股票数据，请检查 ticker。");st.stop()
qd,q=quant(d); f=fundamentals(i); nr,ns=news_engine(news(ticker))
risk=clamp(40+(25 if num(i.get("beta"))>1.7 else 15 if num(i.get("beta"))>1.3 else -10 if num(i.get("beta"))<.8 else 0)+(20 if q["vol"]>.6 else 10 if q["vol"]>.4 else -8 if q["vol"]<.2 else 0))
moat=clamp(50+(15 if f["quality"]>80 else 8 if f["quality"]>65 else -8)+(12 if f["growth"]>75 else 5 if f["growth"]>60 else -8))
hype=clamp(max(q["rsi"]-65,0)*2+max(q["m1"]-.15,0)*100+max(ns-75,0)*.5)
score=clamp(q["score"]*.25+f["score"]*.30+f["valuation"]*.10+moat*.12+ns*.08+(100-risk)*.10+(100-hype)*.05)
verdict="STRONG BUY" if score>=90 else "BUY" if score>=80 else "BUY ON DIPS" if score>=70 else "HOLD" if score>=60 else "WAIT" if score>=50 else "REDUCE / AVOID"

st.markdown(f'<div class="glass"><div class="k">AI-NATIVE EQUITY RESEARCH</div><h1>{ticker} · {i.get("longName",ticker)}</h1><span class="muted">{i.get("sector","")} · {i.get("industry","")}</span></div>',unsafe_allow_html=True)
a,b,c,d1,e=st.columns(5)
a.metric("Price",money(q["price"]));b.metric("Simon Score",f"{score:.0f}");c.metric("Verdict",verdict);d1.metric("Risk",f"{risk:.0f}");e.metric("News",f"{ns:.0f}")
tabs=st.tabs(["✦ Apex AI","📊 Quant","🏢 Fundamentals","📰 Daily News","🧠 Thesis"])
with tabs[0]:
    st.markdown(f'<div class="glass"><div class="k">Simon Score</div><div class="score">{score:.0f}</div><div class="verdict">{verdict}</div><div class="muted">Quant {q["score"]:.0f} · Fundamental {f["score"]:.0f} · Moat {moat:.0f} · News {ns:.0f}</div></div>',unsafe_allow_html=True)
    if run:
        pack={"ticker":ticker,"price":q["price"],"quant":q,"fundamental":f,"moat":moat,"risk":risk,"news_score":ns,
              "news":[{"title":x["title"],"sentiment":x["sentiment"]} for x in nr[:15]],
              "business":i.get("longBusinessSummary","")}
        names=["fundamental","quant","valuation","macro","news","risk"]
        reports={}
        with st.spinner("Running specialist analysts..."):
            for t in names:reports[t]=ai(prompt_for(ticker,pack))
        joined="\n\n".join(f"### {k}\n{v['text']}" for k,v in reports.items())
        bull=ai(f"""You are the Bull Analyst. Find the strongest evidence-based case for owning {ticker}. Then state what evidence would break your thesis.\n{joined}""")
        bear=ai(f"""You are the Bear Analyst. Attack the bullish case for {ticker}. Find the strongest evidence-based reasons not to own it and identify what would invalidate the bear case.\n{joined}""")
        judge=ai(f"""You are Chief Analyst / AI Judge for {ticker}. Adjudicate these reports and Bull/Bear debate. Return:
FINAL VERDICT; SCORE 0-100; CONFIDENCE 0-100; TOP 5 REASONS; TOP 5 RISKS; WHAT CHANGES MY MIND; DATA QUALITY WARNING.
Do not invent a price target or guarantee returns.
SPECIALISTS:
{joined}
BULL:
{bull['text']}
BEAR:
{bear['text']}""")
        st.session_state["apex"]={"reports":reports,"bull":bull,"bear":bear,"judge":judge}
    apex=st.session_state.get("apex")
    if apex:
        st.success("AI Judge: "+str(apex["judge"].get("provider")))
        st.markdown("### ⚖️ Chief Analyst");st.markdown(apex["judge"]["text"])
        st.markdown("### 🐂 Bull Case");st.markdown(apex["bull"]["text"])
        st.markdown("### 🐻 Bear Case");st.markdown(apex["bear"]["text"])
        with st.expander("Specialist reports"):
            for k,v in apex["reports"].items():st.markdown(f"### {k.title()}");st.markdown(v["text"])
    else:st.info("在左侧点击 Run Apex AI。没有 API Key 时仍可使用 Quant/Fundamental/News 引擎。")
with tabs[1]:
    st.line_chart(qd[["Close","MA20","MA50","MA200"]].dropna())
    st.dataframe(qd.tail(40)[["Close","MA20","MA50","MA200","RSI","MACD","ATR","VOL","M3","DD"]],use_container_width=True)
with tabs[2]:
    rows={"Revenue Growth":percent(f["revenue_growth"]),"EPS Growth":percent(f["earnings_growth"]),"Gross Margin":percent(f["gross_margin"]),
          "Operating Margin":percent(f["operating_margin"]),"ROE":percent(f["roe"]),"P/E":f"{f['pe']:.2f}" if not np.isnan(f["pe"]) else "—",
          "Forward P/E":f"{f['forward_pe']:.2f}" if not np.isnan(f["forward_pe"]) else "—","PEG":f"{f['peg']:.2f}" if not np.isnan(f["peg"]) else "—",
          "FCF":money(f["fcf"])}
    st.table(pd.DataFrame({"Metric":list(rows),"Value":list(rows.values())}))
    st.markdown(i.get("longBusinessSummary",""))
with tabs[3]:
    st.metric("News Sentiment",f"{ns:.0f}/100")
    for x in nr[:20]:
        icon="🟢" if x["sentiment"]=="Bullish" else "🔴" if x["sentiment"]=="Bearish" else "🟡"
        st.markdown(f"**{icon} {x['title']}**  \n`{x['publisher']}`")
        if x["url"]:st.markdown(x["url"])
with tabs[4]:
    st.info("V12 首版使用当前会话中的 Thesis Memory。下一 OTA 将加入持久化 thesis、财报事件回测和每日自动重评。")
    if "apex" in st.session_state:
        st.write("Last AI run:",datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
st.divider()
st.caption(f"{VERSION} · Data via Yahoo Finance · AI outputs are research assistance, not financial advice.")