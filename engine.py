import numpy as np
def valuation_snapshot(info,price):
    def n(k):
        try:
            z=float(info.get(k));return z if np.isfinite(z) else np.nan
        except Exception:return np.nan
    pe,fpe,peg,fcf,shares=n("trailingPE"),n("forwardPE"),n("pegRatio"),n("freeCashflow"),n("sharesOutstanding")
    fy=(fcf/(shares*price)) if np.isfinite(fcf) and np.isfinite(shares) and price else np.nan;score=55
    if np.isfinite(peg):score+=20 if peg<1 else 10 if peg<1.5 else -20 if peg>2.5 else 0
    elif np.isfinite(pe):score+=15 if pe<18 else -20 if pe>40 else -10 if pe>30 else 0
    return {"score":float(np.clip(score,0,100)),"pe":"—" if not np.isfinite(pe) else round(pe,2),"forward_pe":"—" if not np.isfinite(fpe) else round(fpe,2),"peg":"—" if not np.isfinite(peg) else round(peg,2),"fcf_yield":"—" if not np.isfinite(fy) else f"{fy*100:.2f}%"}
