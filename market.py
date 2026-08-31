import numpy as np,yfinance as yf
POS=["beat","upgrade","bullish","record","surge","strong","profit","approval","buyback","raised","outperform"]
NEG=["miss","downgrade","bearish","lawsuit","investigation","decline","weak","loss","warning","recall","delay"]
def market_snapshot(h,i):return {"price":float(h["Close"].iloc[-1]),"market_cap":i.get("marketCap","—"),"52w_high":i.get("fiftyTwoWeekHigh","—"),"52w_low":i.get("fiftyTwoWeekLow","—"),"beta":i.get("beta","—")}
def macro_snapshot():
    out={}
    for label,sym in [("S&P 500","^GSPC"),("NASDAQ","^IXIC"),("VIX","^VIX"),("US10Y","^TNX")]:
        try:
            d=yf.Ticker(sym).history(period="5d")
            if not d.empty:
                a=float(d["Close"].iloc[-1]);b=float(d["Close"].iloc[-2]) if len(d)>1 else a
                out[label]={"value":a,"change":a/b-1 if b else 0}
        except Exception:pass
    return out
def news_snapshot(raw):
    rows=[];scores=[]
    for item in raw:
        c=item.get("content",item);title=str(c.get("title") or item.get("title") or "").strip()
        if not title:continue
        low=title.lower();s=sum(w in low for w in POS)-sum(w in low for w in NEG);scores.append(s)
        p=c.get("provider");pub=p.get("displayName") if isinstance(p,dict) else "Yahoo Finance"
        rows.append({"title":title,"publisher":pub,"sentiment":"Bullish" if s>0 else "Bearish" if s<0 else "Neutral"})
    return rows,float(np.clip(50+(np.mean(scores)*15 if scores else 0),0,100))
