import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import math
import json
from datetime import datetime, timezone

# ============================================================
# SIMON STOCK V5.0 ULTIMATE
# AI INVESTMENT OS
# ============================================================

st.set_page_config(
    page_title="Simon Stock V5",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLE
# ============================================================

st.markdown("""
<style>
.block-container {
    max-width: 1450px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 28px;
    border-radius: 24px;
    margin-bottom: 22px;
    border: 1px solid rgba(128,128,128,.22);
    background: linear-gradient(
        135deg,
        rgba(80,100,180,.12),
        rgba(140,80,180,.08)
    );
}

.hero-title {
    font-size: 42px;
    font-weight: 850;
    margin-bottom: 0;
}

.hero-subtitle {
    font-size: 17px;
    opacity: .72;
    margin-top: 5px;
}

.card {
    padding: 20px;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,.18);
    background: rgba(128,128,128,.055);
    margin-bottom: 14px;
}

.score {
    font-size: 54px;
    font-weight: 900;
    line-height: 1;
}

.verdict {
    font-size: 27px;
    font-weight: 800;
}

.muted {
    opacity: .65;
    font-size: 13px;
}

.section {
    font-size: 24px;
    font-weight: 800;
    margin-top: 15px;
    margin-bottom: 10px;
}

div[data-testid="stMetric"] {
    border-radius: 14px;
}

.stButton button {
    border-radius: 12px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=np.nan):
    try:
        if value is None:
            return default
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def fmt_money(value):
    value = safe_float(value)
    if np.isnan(value):
        return "N/A"

    if abs(value) >= 1e12:
        return f"${value / 1e12:.2f}T"

    if abs(value) >= 1e9:
        return f"${value / 1e9:.2f}B"

    if abs(value) >= 1e6:
        return f"${value / 1e6:.2f}M"

    return f"${value:,.2f}"


def fmt_percent(value):
    value = safe_float(value)

    if np.isnan(value):
        return "N/A"

    return f"{value * 100:.1f}%"


def clean_text(value, fallback="N/A"):
    if value is None:
        return fallback

    text = str(value).strip()

    if not text or text.lower() in ["nan", "none"]:
        return fallback

    return text


def get_secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return default


# ============================================================
# MARKET DATA
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def load_market_data(ticker, period):
    ticker = ticker.upper().strip()

    stock = yf.Ticker(ticker)

    history = stock.history(
        period=period,
        interval="1d",
        auto_adjust=False
    )

    try:
        info = stock.info
    except Exception:
        info = {}

    try:
        news = stock.news
    except Exception:
        news = []

    return history, info, news


@st.cache_data(ttl=900, show_spinner=False)
def load_financials(ticker):
    stock = yf.Ticker(ticker)

    try:
        income = stock.income_stmt
    except Exception:
        income = pd.DataFrame()

    try:
        balance = stock.balance_sheet
    except Exception:
        balance = pd.DataFrame()

    try:
        cashflow = stock.cashflow
    except Exception:
        cashflow = pd.DataFrame()

    return income, balance, cashflow


# ============================================================
# TECHNICAL ANALYSIS
# ============================================================

def technical_analysis(history):

    if history is None or history.empty:
        return {}

    close = history["Close"].dropna()

    if len(close) < 20:
        return {}

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    latest = safe_float(close.iloc[-1])

    result = {
        "price": latest,
        "sma20": safe_float(sma20.iloc[-1]),
        "sma50": safe_float(sma50.iloc[-1]),
    }

    if len(close) >= 200:
        sma200 = close.rolling(200).mean()
        result["sma200"] = safe_float(sma200.iloc[-1])
    else:
        result["sma200"] = np.nan

    if len(close) >= 15:

        delta = close.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)

        rsi = 100 - (
            100 / (1 + rs)
        )

        result["rsi"] = safe_float(rsi.iloc[-1])

    else:
        result["rsi"] = np.nan

    if len(close) >= 30:
        result["return_30d"] = (
            close.iloc[-1] /
            close.iloc[-30] - 1
        )
    else:
        result["return_30d"] = np.nan

    if len(close) >= 252:
        result["return_1y"] = (
            close.iloc[-1] /
            close.iloc[-252] - 1
        )
    else:
        result["return_1y"] = np.nan

    return result


# ============================================================
# SIMON SCORE
# ============================================================

def calculate_simon_score(info):

    score = 50

    details = {
        "Business": 50,
        "Growth": 50,
        "Profitability": 50,
        "Financial": 50,
        "Valuation": 50,
        "Risk": 50,
    }

    # Growth

    revenue_growth = safe_float(
        info.get("revenueGrowth")
    )

    earnings_growth = safe_float(
        info.get("earningsGrowth")
    )

    if not np.isnan(revenue_growth):

        if revenue_growth >= .20:
            score += 8
            details["Growth"] += 25

        elif revenue_growth >= .10:
            score += 5
            details["Growth"] += 15

        elif revenue_growth < 0:
            score -= 6
            details["Growth"] -= 20

    if not np.isnan(earnings_growth):

        if earnings_growth >= .20:
            score += 8
            details["Growth"] += 20

        elif earnings_growth >= .10:
            score += 4

        elif earnings_growth < 0:
            score -= 6
            details["Growth"] -= 15

    # ROE

    roe = safe_float(
        info.get("returnOnEquity")
    )

    if not np.isnan(roe):

        if roe >= .25:
            score += 10
            details["Profitability"] += 30

        elif roe >= .15:
            score += 6
            details["Profitability"] += 18

        elif roe < .05:
            score -= 4
            details["Profitability"] -= 15

    # Margin

    margin = safe_float(
        info.get("profitMargins")
    )

    if not np.isnan(margin):

        if margin >= .25:
            score += 8
            details["Profitability"] += 20

        elif margin >= .15:
            score += 4

        elif margin < 0:
            score -= 8
            details["Profitability"] -= 20

    # FCF

    fcf = safe_float(
        info.get("freeCashflow")
    )

    if not np.isnan(fcf):

        if fcf > 0:
            score += 7
            details["Financial"] += 20

        else:
            score -= 7
            details["Financial"] -= 20

    # Debt

    debt_to_equity = safe_float(
        info.get("debtToEquity")
    )

    if not np.isnan(debt_to_equity):

        if debt_to_equity < 50:
            score += 5
            details["Financial"] += 15

        elif debt_to_equity > 200:
            score -= 8
            details["Financial"] -= 20

    # PE

    pe = safe_float(
        info.get("trailingPE")
    )

    if not np.isnan(pe) and pe > 0:

        if pe < 18:
            score += 8
            details["Valuation"] += 25

        elif pe < 25:
            score += 4
            details["Valuation"] += 12

        elif pe > 50:
            score -= 10
            details["Valuation"] -= 25

        elif pe > 35:
            score -= 6
            details["Valuation"] -= 15

    # Beta

    beta = safe_float(
        info.get("beta")
    )

    if not np.isnan(beta):

        if beta > 2:
            score -= 4
            details["Risk"] -= 15

        elif beta < 1:
            details["Risk"] += 10

    # Business size / maturity

    market_cap = safe_float(
        info.get("marketCap")
    )

    if not np.isnan(market_cap):

        if market_cap > 100e9:
            details["Business"] += 20

        elif market_cap > 10e9:
            details["Business"] += 10

    for key in details:
        details[key] = max(
            0,
            min(
                100,
                details[key]
            )
        )

    score = max(
        0,
        min(
            100,
            int(round(score))
        )
    )

    return score, details


# ============================================================
# VALUATION ENGINE
# ============================================================

def calculate_valuation(info, price):

    price = safe_float(price)

    pe = safe_float(
        info.get("trailingPE")
    )

    forward_pe = safe_float(
        info.get("forwardPE")
    )

    earnings_growth = safe_float(
        info.get("earningsGrowth")
    )

    revenue_growth = safe_float(
        info.get("revenueGrowth")
    )

    fair_values = []

    # Historical / normalized PE approach

    if not np.isnan(pe) and pe > 0:

        if pe <= 18:
            normalized_multiple = 22
        elif pe <= 25:
            normalized_multiple = 24
        elif pe <= 35:
            normalized_multiple = 27
        elif pe <= 50:
            normalized_multiple = 30
        else:
            normalized_multiple = 32

        pe_fair = price * (
            normalized_multiple / pe
        )

        fair_values.append(
            pe_fair
        )

    # Forward PE

    if (
        not np.isnan(forward_pe)
        and forward_pe > 0
    ):

        forward_fair = price * (
            24 / forward_pe
        )

        fair_values.append(
            forward_fair
        )

    # Growth adjustment

    if fair_values:

        fair = float(
            np.median(fair_values)
        )

    else:
        fair = price

    growth_adjustment = 1.0

    if not np.isnan(earnings_growth):

        if earnings_growth > .25:
            growth_adjustment += .08

        elif earnings_growth > .15:
            growth_adjustment += .04

        elif earnings_growth < 0:
            growth_adjustment -= .10

    if not np.isnan(revenue_growth):

        if revenue_growth > .20:
            growth_adjustment += .03

        elif revenue_growth < 0:
            growth_adjustment -= .05

    fair *= growth_adjustment

    fair = max(
        price * .40,
        min(
            price * 2.0,
            fair
        )
    )

    return {
        "strong_buy": fair * .70,
        "buy": fair * .82,
        "fair": fair,
        "bull": fair * 1.25,
        "bear": fair * .55,
    }


# ============================================================
# VERDICT
# ============================================================

def get_verdict(
    score,
    price,
    valuation
):

    fair = valuation["fair"]
    strong_buy = valuation["strong_buy"]
    buy = valuation["buy"]

    upside = (
        fair / price - 1
    ) if price > 0 else 0

    if (
        score >= 85
        and price <= strong_buy
    ):
        return "🟢 STRONG BUY", upside

    if (
        score >= 75
        and price <= buy
    ):
        return "🟢 BUY", upside

    if (
        score >= 70
        and price <= fair * 1.08
    ):
        return "🟡 WATCH / ACCUMULATE", upside

    if score >= 60:
        return "🟠 WATCH", upside

    return "🔴 AVOID / RESEARCH", upside


# ============================================================
# NEWS
# ============================================================

def normalize_news(news):

    rows = []

    if not isinstance(news, list):
        return pd.DataFrame()

    for item in news:

        try:

            content = item.get(
                "content",
                item
            )

            title = (
                content.get("title")
                or item.get("title")
            )

            publisher = (
                content.get("provider", {})
                .get("displayName")
                if isinstance(
                    content.get("provider"),
                    dict
                )
                else None
            )

            if not publisher:
                publisher = (
                    item.get("publisher")
                )

            url = (
                content.get("canonicalUrl", {})
                .get("url")
                if isinstance(
                    content.get("canonicalUrl"),
                    dict
                )
                else None
            )

            if not url:
                url = item.get("link")

            if title:

                rows.append({
                    "title": title,
                    "publisher": publisher or "Unknown",
                    "url": url or ""
                })

        except Exception:
            continue

    return pd.DataFrame(rows)


# ============================================================
# OPENAI
# ============================================================

def get_openai_client():

    api_key = get_secret(
        "OPENAI_API_KEY"
    )

    if not api_key:
        return None

    try:
        from openai import OpenAI

        return OpenAI(
            api_key=api_key
        )

    except Exception:
        return None


def run_ai_analysis(
    ticker,
    info,
    price,
    score,
    valuation,
    risk,
    horizon,
):

    client = get_openai_client()

    if client is None:

        return None, (
            "OPENAI_API_KEY 尚未配置。"
        )

    model = get_secret(
        "OPENAI_MODEL",
        "gpt-5"
    )

    company = clean_text(
        info.get(
            "longName",
            ticker
        ),
        ticker
    )

    data = {

        "ticker": ticker,

        "company": company,

        "price": price,

        "simon_score": score,

        "PE": info.get(
            "trailingPE"
        ),

        "forward_PE": info.get(
            "forwardPE"
        ),

        "ROE": info.get(
            "returnOnEquity"
        ),

        "profit_margin": info.get(
            "profitMargins"
        ),

        "revenue_growth": info.get(
            "revenueGrowth"
        ),

        "earnings_growth": info.get(
            "earningsGrowth"
        ),

        "free_cash_flow": info.get(
            "freeCashflow"
        ),

        "debt_to_equity": info.get(
            "debtToEquity"
        ),

        "market_cap": info.get(
            "marketCap"
        ),

        "strong_buy": valuation[
            "strong_buy"
        ],

        "buy": valuation[
            "buy"
        ],

        "fair_value": valuation[
            "fair"
        ],

        "bull_value": valuation[
            "bull"
        ],

        "bear_value": valuation[
            "bear"
        ],

        "risk_profile": risk,

        "horizon": horizon,
    }

    prompt = f"""
You are Simon Stock, an advanced AI
investment research assistant.

You are NOT a financial adviser.
Do not guarantee returns.
Do not pretend to be Warren Buffett,
Charlie Munger, 段永平, Peter Lynch,
or Philip Fisher.

Instead, use publicly known investment
principles associated with their
approaches as analytical frameworks.

Analyze this company:

{json.dumps(data, ensure_ascii=False, indent=2)}

Your job is NOT to predict tomorrow's
stock price.

Your job is to determine whether the
CURRENT PRICE offers an attractive
risk/reward relationship.

Use this structure:

# 🧠 Simon Verdict

Choose exactly one:

STRONG BUY
BUY
WATCH
AVOID

Then explain why.

# 1. What the company actually does

Explain the business in simple language.

# 2. Business Quality

Evaluate:
- economics
- recurring revenue
- pricing power
- scalability
- capital intensity

# 3. Moat

Evaluate:
- brand
- network effects
- switching costs
- cost advantage
- distribution
- data
- intellectual property

Do not invent a moat.

# 4. Management

Evaluate capital allocation,
incentives and execution.

Say when data is unavailable.

# 5. Buffett Lens

Ask:

Would this still be a good business
if the stock market closed for five years?

# 6. Munger Lens

Use inversion.

What could permanently destroy
shareholder value?

# 7. 段永平 Lens

Evaluate:

Right Business
Right People
Right Price

# 8. Lynch Lens

Evaluate growth versus valuation.

# 9. Fisher Lens

Evaluate long-term growth potential,
R&D, product strength and execution.

# 10. Financial Quality

Discuss:
- revenue
- margins
- ROE
- FCF
- debt
- earnings

Separate facts from assumptions.

# 11. Valuation

Use the supplied valuation estimates.

Explain why the assumptions may be
too optimistic or too pessimistic.

# 12. Bull Case

Give the strongest reasonable case.

# 13. Base Case

Give the most reasonable case.

# 14. Bear Case

Give the strongest reasonable downside case.

# 15. Devil's Advocate

Attack your own conclusion.

Give at least 5 reasons
the thesis could be wrong.

# 16. Simon Buy Zone

Explain the difference between:

Strong Buy
Buy
Fair
Expensive

Do not pretend these prices are precise.

# 17. Biggest Unknown

What information would most change
your conclusion?

# 18. Final Score

Give:

Business Quality /100
Moat /100
Management /100
Growth /100
Financial Quality /100
Valuation /100
Risk /100
Overall /100

Important:

Do not fabricate facts.
If information is missing,
say "insufficient data".

The goal is disciplined research,
not persuasion.
"""

    try:

        response = client.responses.create(
            model=model,
            input=prompt
        )

        return response.output_text, None

    except Exception as e:

        return None, (
            "GPT 调用失败："
            + str(e)
        )


# ============================================================
# PORTFOLIO
# ============================================================

def parse_portfolio(text):

    rows = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        parts = [
            x.strip()
            for x in line.split(",")
        ]

        if len(parts) < 3:
            continue

        try:

            ticker = parts[0].upper()

            shares = float(parts[1])

            cost = float(parts[2])

            history, info, _ = load_market_data(
                ticker,
                "5d"
            )

            if history.empty:
                continue

            price = safe_float(
                history["Close"].iloc[-1]
            )

            value = price * shares

            invested = cost * shares

            pnl = value - invested

            pnl_pct = (
                pnl / invested * 100
                if invested
                else 0
            )

            rows.append({

                "Ticker": ticker,

                "Shares": shares,

                "Cost": cost,

                "Price": round(
                    price,
                    2
                ),

                "Value": round(
                    value,
                    2
                ),

                "P/L": round(
                    pnl,
                    2
                ),

                "P/L %": round(
                    pnl_pct,
                    2
                )

            })

        except Exception:
            continue

    return pd.DataFrame(rows)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<div class="muted">
SIMON STOCK V5.0 ULTIMATE
</div>

<div class="hero-title">
🧠 Simon Stock
</div>

<div class="hero-subtitle">
AI Investment OS · Think Like an Owner · Invest With a Plan
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🔎 Research")

    ticker = st.text_input(
        "股票代码",
        value="AAPL",
        max_chars=15
    ).upper().strip()

    period = st.selectbox(
        "历史数据",
        [
            "1mo",
            "3mo",
            "6mo",
            "1y",
            "2y",
            "5y",
            "10y"
        ],
        index=3
    )

    st.divider()

    st.header("🧬 Your Investment DNA")

    risk = st.selectbox(
        "风险偏好",
        [
            "保守",
            "平衡",
            "进取"
        ],
        index=1
    )

    horizon = st.selectbox(
        "投资期限",
        [
            "< 1年",
            "1–3年",
            "3–5年",
            "5年以上"
        ],
        index=2
    )

    st.divider()

    st.caption(
        "Simon Stock V5.0"
    )

    st.caption(
        "Investment research only."
    )


# ============================================================
# LOAD DATA
# ============================================================

if not ticker:

    st.warning(
        "请输入股票代码。"
    )

    st.stop()


with st.spinner(
    f"正在读取 {ticker} 数据..."
):

    try:

        history, info, news = load_market_data(
            ticker,
            period
        )

    except Exception as e:

        st.error(
            "股票数据读取失败。"
        )

        st.code(
            str(e)
        )

        st.stop()


if history is None or history.empty:

    st.error(
        f"找不到 {ticker} 的有效市场数据。"
    )

    st.info(
        "请检查代码，例如 AAPL、GOOGL、MSFT、"
        "NVDA、AVGO、BABA、PDD、TSM。"
    )

    st.stop()


price = safe_float(
    history["Close"].iloc[-1]
)

previous = safe_float(
    history["Close"].iloc[-2]
    if len(history) >= 2
    else np.nan
)

daily_change = (
    price / previous - 1
) if (
    not np.isnan(previous)
    and previous != 0
) else np.nan

company = clean_text(
    info.get(
        "longName",
        ticker
    ),
    ticker
)

score, score_details = calculate_simon_score(
    info
)

valuation = calculate_valuation(
    info,
    price
)

verdict, upside = get_verdict(
    score,
    price,
    valuation
)

technical = technical_analysis(
    history
)

news_df = normalize_news(
    news
)


# ============================================================
# TOP METRICS
# ============================================================

st.subheader(
    f"{company} · {ticker}"
)

c1, c2, c3, c4, c5 = st.columns(5)

with c1:

    st.metric(
        "Price",
        f"${price:.2f}",
        (
            f"{daily_change * 100:+.2f}%"
            if not np.isnan(daily_change)
            else None
        )
    )

with c2:

    st.metric(
        "Simon Score",
        f"{score}/100"
    )

with c3:

    pe = safe_float(
        info.get("trailingPE")
    )

    st.metric(
        "P/E",
        (
            f"{pe:.1f}x"
            if not np.isnan(pe)
            else "N/A"
        )
    )

with c4:

    roe = safe_float(
        info.get("returnOnEquity")
    )

    st.metric(
        "ROE",
        fmt_percent(roe)
    )

with c5:

    market_cap = safe_float(
        info.get("marketCap")
    )

    st.metric(
        "Market Cap",
        fmt_money(market_cap)
    )


# ============================================================
# MAIN TABS
# ============================================================

tabs = st.tabs([
    "🏠 Dashboard",
    "🤖 Simon AI",
    "🏆 Master Council",
    "💰 Valuation",
    "📊 Fundamentals",
    "📰 News",
    "⚔️ Battle",
    "💼 Portfolio",
])


# ============================================================
# DASHBOARD
# ============================================================

with tabs[0]:

    st.markdown(
        '<div class="section">🧠 Simon Verdict</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(
        [1, 2]
    )

    with left:

        st.markdown(
            f"""
            <div class="card">

            <div class="muted">
            SIMON SCORE
            </div>

            <div class="score">
            {score}
            </div>

            <div class="muted">
            /100
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with right:

        st.markdown(
            f"""
            <div class="card">

            <div class="muted">
            CURRENT VERDICT
            </div>

            <div class="verdict">
            {verdict}
            </div>

            <p>
            Fair Value Upside:
            <b>
            {upside * 100:+.1f}%
            </b>
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section">📈 Price</div>',
        unsafe_allow_html=True
    )

    chart_df = history[
        ["Close"]
    ].rename(
        columns={
            "Close": ticker
        }
    )

    st.line_chart(
        chart_df,
        height=430
    )

    st.markdown(
        '<div class="section">🎯 Simon Price Map</div>',
        unsafe_allow_html=True
    )

    p1, p2, p3, p4, p5 = st.columns(5)

    p1.metric(
        "🟢 Strong Buy",
        f"${valuation['strong_buy']:.2f}"
    )

    p2.metric(
        "🟢 Buy",
        f"${valuation['buy']:.2f}"
    )

    p3.metric(
        "🟡 Fair Value",
        f"${valuation['fair']:.2f}"
    )

    p4.metric(
        "🟠 Bull",
        f"${valuation['bull']:.2f}"
    )

    p5.metric(
        "🔴 Bear",
        f"${valuation['bear']:.2f}"
    )

    st.markdown(
        '<div class="section">📊 Simon Dimensions</div>',
        unsafe_allow_html=True
    )

    score_df = pd.DataFrame(
        {
            "Dimension": list(
                score_details.keys()
            ),
            "Score": list(
                score_details.values()
            )
        }
    )

    st.dataframe(
        score_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section">📐 Technical Snapshot</div>',
        unsafe_allow_html=True
    )

    tc1, tc2, tc3, tc4 = st.columns(4)

    with tc1:

        st.metric(
            "SMA 20",
            (
                f"${technical['sma20']:.2f}"
                if "sma20" in technical
                and not np.isnan(
                    technical["sma20"]
                )
                else "N/A"
            )
        )

    with tc2:

        st.metric(
            "SMA 50",
            (
                f"${technical['sma50']:.2f}"
                if "sma50" in technical
                and not np.isnan(
                    technical["sma50"]
                )
                else "N/A"
            )
        )

    with tc3:

        rsi = technical.get(
            "rsi",
            np.nan
        )

        st.metric(
            "RSI",
            (
                f"{rsi:.1f}"
                if not np.isnan(rsi)
                else "N/A"
            )
        )

    with tc4:

        ret = technical.get(
            "return_1y",
            np.nan
        )

        st.metric(
            "1Y Return",
            (
                f"{ret * 100:+.1f}%"
                if not np.isnan(ret)
                else "N/A"
            )
        )


# ============================================================
# AI
# ============================================================

with tabs[1]:

    st.subheader(
        "🤖 Simon Intelligence"
    )

    st.write(
        "不是预测明天涨跌，而是研究："
    )

    st.markdown(
        """
        > **如果这是我的钱，我为什么现在要买？**
        >
        > **如果我不该买，真正的原因是什么？**
        """
    )

    ai_ready = bool(
        get_secret(
            "OPENAI_API_KEY"
        )
    )

    if ai_ready:

        st.success(
            "🟢 Simon AI 已连接"
        )

    else:

        st.warning(
            "🟡 Simon AI 尚未配置 API Key。"
            "基础股票分析仍可使用。"
        )

    if st.button(
        "🚀 Run Simon Deep Analysis",
        type="primary",
        use_container_width=True,
        disabled=not ai_ready
    ):

        with st.spinner(
            "Simon 正在进行多框架投资研究..."
        ):

            result, error = run_ai_analysis(
                ticker,
                info,
                price,
                score,
                valuation,
                risk,
                horizon
            )

        if error:

            st.error(
                error
            )

        else:

            st.markdown(
                result
            )

    st.divider()

    st.markdown(
        """
        ### Simon AI 的核心流程

        **Business → Moat → Management → Growth → Financials → Valuation → Risk → Devil's Advocate → Verdict**
        """
    )


# ============================================================
# MASTER COUNCIL
# ============================================================

with tabs[2]:

    st.subheader(
        "🏆 Investment Master Council"
    )

    st.caption(
        "这里不是模拟真人发言，而是使用公开的投资思想作为分析框架。"
    )

    council = {

        "🧓 Buffett":
        """
        **核心问题：**

        如果股市关闭五年，我还愿意持有这家公司吗？

        关注：
        - 护城河
        - 长期现金流
        - 定价权
        - 管理层
        - 资本配置
        - 长期竞争优势
        """,

        "🧠 Charlie Munger":
        """
        **核心问题：**

        什么东西可能让我永久亏钱？

        使用反向思维：
        - 杠杆
        - 激励机制
        - 竞争
        - 会计风险
        - 周期
        - 管理错误
        """,

        "🎮 段永平":
        """
        **核心问题：**

        Right Business？
        Right People？
        Right Price？

        特别强调：
        - 是否真正理解生意
        - 商业模式
        - 管理层诚信
        - 长期价值
        """,

        "🔍 Peter Lynch":
        """
        **核心问题：**

        增长是否值得当前估值？

        关注：
        - EPS 增长
        - PEG
        - 消费者需求
        - 产品
        - 行业空间
        """,

        "🔬 Philip Fisher":
        """
        **核心问题：**

        这家公司十年以后会不会比今天强很多？

        关注：
        - R&D
        - 产品
        - 销售
        - 管理
        - 长期成长
        """
    }

    for name, content in council.items():

        with st.expander(
            name,
            expanded=True
        ):

            st.markdown(
                content
            )

    st.divider()

    st.subheader(
        "⚔️ Council Debate"
    )

    st.info(
        "真正的旗舰版下一阶段可以让 GPT 根据实际财务数据，让这些框架互相辩论，而不是只显示固定文字。"
    )


# ============================================================
# VALUATION
# ============================================================

with tabs[3]:

    st.subheader(
        "💰 Simon Valuation Lab"
    )

    st.warning(
        "估值模型是情景分析，不是精确预测。"
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Strong Buy",
        f"${valuation['strong_buy']:.2f}"
    )

    c2.metric(
        "Buy",
        f"${valuation['buy']:.2f}"
    )

    c3.metric(
        "Fair Value",
        f"${valuation['fair']:.2f}"
    )

    c4.metric(
        "Bull",
        f"${valuation['bull']:.2f}"
    )

    c5.metric(
        "Bear",
        f"${valuation['bear']:.2f}"
    )

    st.divider()

    if price <= valuation["strong_buy"]:

        st.success(
            "🟢 当前进入 Strong Buy 估值区"
        )

    elif price <= valuation["buy"]:

        st.success(
            "🟢 当前进入 Buy 估值区"
        )

    elif price <= valuation["fair"]:

        st.warning(
            "🟡 当前接近合理价值"
        )

    elif price <= valuation["bull"]:

        st.warning(
            "🟠 当前已经偏贵"
        )

    else:

        st.error(
            "🔴 当前价格明显高于模型合理区间"
        )

    st.markdown(
        """
        ### Simon 的估值哲学

        **好公司 ≠ 好价格**

        一家公司可以：

        > 商业模式优秀  
        > 护城河很深  
        > 管理层优秀  
        > 长期增长强

        但如果价格太贵：

        **Simon 仍然可以选择 WAIT。**
        """
    )

    pe = safe_float(
        info.get("trailingPE")
    )

    forward_pe = safe_float(
        info.get("forwardPE")
    )

    growth = safe_float(
        info.get("earningsGrowth")
    )

    vc1, vc2, vc3 = st.columns(3)

    vc1.metric(
        "Trailing PE",
        (
            f"{pe:.1f}x"
            if not np.isnan(pe)
            else "N/A"
        )
    )

    vc2.metric(
        "Forward PE",
        (
            f"{forward_pe:.1f}x"
            if not np.isnan(forward_pe)
            else "N/A"
        )
    )

    vc3.metric(
        "Earnings Growth",
        fmt_percent(growth)
    )


# ============================================================
# FUNDAMENTALS
# ============================================================

with tabs[4]:

    st.subheader(
        "📊 Fundamental Intelligence"
    )

    income, balance, cashflow = load_financials(
        ticker
    )

    metrics = {

        "Revenue Growth":
            fmt_percent(
                info.get(
                    "revenueGrowth"
                )
            ),

        "Earnings Growth":
            fmt_percent(
                info.get(
                    "earningsGrowth"
                )
            ),

        "Profit Margin":
            fmt_percent(
                info.get(
                    "profitMargins"
                )
            ),

        "Operating Margin":
            fmt_percent(
                info.get(
                    "operatingMargins"
                )
            ),

        "ROE":
            fmt_percent(
                info.get(
                    "returnOnEquity"
                )
            ),

        "ROA":
            fmt_percent(
                info.get(
                    "returnOnAssets"
                )
            ),

        "Debt / Equity":
            (
                f"{safe_float(info.get('debtToEquity')):.1f}"
                if not np.isnan(
                    safe_float(
                        info.get(
                            "debtToEquity"
                        )
                    )
                )
                else "N/A"
            ),

        "Free Cash Flow":
            fmt_money(
                info.get(
                    "freeCashflow"
                )
            ),

        "Operating Cash Flow":
            fmt_money(
                info.get(
                    "operatingCashflow"
                )
            ),

        "Dividend Yield":
            fmt_percent(
                info.get(
                    "dividendYield"
                )
            ),
    }

    rows = []

    for k, v in metrics.items():

        rows.append({
            "Metric": k,
            "Value": v
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    if not income.empty:

        st.subheader(
            "Income Statement"
        )

        st.dataframe(
            income.head(10),
            use_container_width=True
        )

    if not cashflow.empty:

        st.subheader(
            "Cash Flow"
        )

        st.dataframe(
            cashflow.head(10),
            use_container_width=True
        )


# ============================================================
# NEWS
# ============================================================

with tabs[5]:

    st.subheader(
        "📰 News Intelligence"
    )

    if news_df.empty:

        st.info(
            "暂时没有读取到新闻。"
        )

    else:

        for _, row in news_df.head(10).iterrows():

            title = row["title"]

            publisher = row["publisher"]

            url = row["url"]

            st.markdown(
                f"### {title}"
            )

            st.caption(
                f"Source: {publisher}"
            )

            if url:

                st.link_button(
                    "阅读原文",
                    url
                )

            st.divider()


# ============================================================
# BATTLE
# ============================================================

with tabs[6]:

    st.subheader(
        "⚔️ Simon Stock Battle"
    )

    battle_input = st.text_input(
        "输入 2–4 个股票代码",
        "AAPL,GOOGL,AVGO"
    )

    if st.button(
        "⚔️ Start Battle",
        type="primary"
    ):

        symbols = [
            x.strip().upper()
            for x in battle_input.split(",")
            if x.strip()
        ]

        symbols = list(
            dict.fromkeys(
                symbols
            )
        )[:4]

        results = []

        with st.spinner(
            "Simon 正在比较..."
        ):

            for symbol in symbols:

                try:

                    h, inf, _ = load_market_data(
                        symbol,
                        "1y"
                    )

                    if h.empty:
                        continue

                    p = safe_float(
                        h["Close"].iloc[-1]
                    )

                    s, details = calculate_simon_score(
                        inf
                    )

                    val = calculate_valuation(
                        inf,
                        p
                    )

                    v, up = get_verdict(
                        s,
                        p,
                        val
                    )

                    results.append({

                        "Ticker": symbol,

                        "Price": round(
                            p,
                            2
                        ),

                        "Simon Score": s,

                        "Fair Value": round(
                            val["fair"],
                            2
                        ),

                        "Upside": round(
                            up * 100,
                            1
                        ),

                        "Verdict": v

                    })

                except Exception:
                    continue

        if not results:

            st.error(
                "没有成功读取 Battle 数据。"
            )

        else:

            battle_df = pd.DataFrame(
                results
            ).sort_values(
                "Simon Score",
                ascending=False
            )

            st.dataframe(
                battle_df,
                use_container_width=True,
                hide_index=True
            )

            winner = battle_df.iloc[0]

            st.success(
                f"🏆 Simon Winner: "
                f"{winner['Ticker']} · "
                f"{winner['Simon Score']}/100"
            )


# ============================================================
# PORTFOLIO
# ============================================================

with tabs[7]:

    st.subheader(
        "💼 Simon Portfolio Brain"
    )

    st.write(
        """
        输入：

        `Ticker, 股数, 成本价`
        """
    )

    portfolio_text = st.text_area(
        "Portfolio",
        """AAPL,2,310
GOOGL,2,342
AVGO,2,352""",
        height=160
    )

    if st.button(
        "🧠 Analyze My Portfolio",
        type="primary"
    ):

        portfolio_df = parse_portfolio(
            portfolio_text
        )

        if portfolio_df.empty:

            st.error(
                "没有成功读取持仓。"
            )

        else:

            st.dataframe(
                portfolio_df,
                use_container_width=True,
                hide_index=True
            )

            total_value = portfolio_df[
                "Value"
            ].sum()

            total_pnl = portfolio_df[
                "P/L"
            ].sum()

            total_invested = (
                portfolio_df[
                    "Cost"
                ]
                * portfolio_df[
                    "Shares"
                ]
            ).sum()

            total_return = (
                total_pnl /
                total_invested * 100
                if total_invested
                else 0
            )

            pc1, pc2, pc3 = st.columns(3)

            pc1.metric(
                "Portfolio Value",
                f"${total_value:,.2f}"
            )

            pc2.metric(
                "Total P/L",
                f"${total_pnl:,.2f}"
            )

            pc3.metric(
                "Return",
                f"{total_return:+.2f}%"
            )

            st.divider()

            st.subheader(
                "🎯 Simon Portfolio Diagnosis"
            )

            if len(portfolio_df) == 1:

                st.warning(
                    "⚠️ 组合高度集中在单一股票。"
                )

            elif len(portfolio_df) <= 3:

                st.warning(
                    "🟡 持仓数量较少，单股风险较高。"
                )

            else:

                st.success(
                    "🟢 持仓数量相对分散。"
                )

            best = portfolio_df.loc[
                portfolio_df["P/L"].idxmax()
            ]

            worst = portfolio_df.loc[
                portfolio_df["P/L"].idxmin()
            ]

            st.write(
                f"表现最好：**{best['Ticker']}**"
            )

            st.write(
                f"表现最差：**{worst['Ticker']}**"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; opacity:.55;">

    🧠 <b>SIMON STOCK V5.0 ULTIMATE</b>

    <br>

    AI Investment OS

    <br><br>

    Think Like an Owner. Invest With a Plan.

    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "Simon Stock is an investment research and education tool. "
    "It does not provide guaranteed returns or personalized financial advice."
)