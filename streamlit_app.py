import os
import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ============================================================
# SIMON STOCK V11 AI ULTIMATE
# First Principles + Quant + Fundamental + Valuation
# Daily News + Anti-Hype + Moat + Risk/Reward
# Bull / Bear / AI Judge + Portfolio + Watchlist
# ============================================================

st.set_page_config(
    page_title="Simon Stock V11 AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------

DEFAULT_WATCHLIST = [
    "AAPL",
    "GOOGL",
    "NVDA",
    "MSFT",
    "AMZN",
    "META",
    "AVGO",
    "TSM",
    "QCOM",
    "PDD",
]

MARKET_SYMBOLS = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Dow Jones": "^DJI",
    "Russell 2000": "^RUT",
    "VIX": "^VIX",
}

POSITIVE_WORDS = [
    "beat",
    "beats",
    "upgrade",
    "upgraded",
    "bullish",
    "growth",
    "record",
    "surge",
    "strong",
    "profit",
    "approval",
    "buyback",
    "raises",
    "raised",
    "outperform",
    "partnership",
]

NEGATIVE_WORDS = [
    "miss",
    "misses",
    "downgrade",
    "downgraded",
    "bearish",
    "lawsuit",
    "investigation",
    "decline",
    "weak",
    "loss",
    "cut",
    "warning",
    "recall",
    "layoff",
    "layoffs",
    "fraud",
    "delay",
]

# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------

if "watchlist" not in st.session_state:
    st.session_state.watchlist = DEFAULT_WATCHLIST.copy()

if "portfolio" not in st.session_state:
    st.session_state.portfolio = []

if "last_ai_report" not in st.session_state:
    st.session_state.last_ai_report = ""

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def safe_float(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return np.nan


def fmt_money(value):
    value = safe_float(value)
    if np.isnan(value):
        return "—"
    return f"${value:,.2f}"


def fmt_pct(value):
    value = safe_float(value)
    if np.isnan(value):
        return "—"
    return f"{value * 100:.1f}%"


def clamp(value, low=0, high=100):
    return float(np.clip(value, low, high))


def grade(score):
    if score >= 90:
        return "S"
    if score >= 80:
        return "A"
    if score >= 70:
        return "B"
    if score >= 60:
        return "C"
    if score >= 50:
        return "D"
    return "E"


def action(score):
    if score >= 90:
        return "🔥 STRONG BUY"
    if score >= 80:
        return "🟢 BUY"
    if score >= 70:
        return "🟢 BUY ON DIPS"
    if score >= 60:
        return "🔵 HOLD"
    if score >= 50:
        return "🟡 WAIT"
    return "🔴 REDUCE / AVOID"


# ------------------------------------------------------------
# DATA ENGINE
# ------------------------------------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def get_history(ticker, period="1y", interval="1d"):
    try:
        obj = yf.Ticker(ticker.upper())
        data = obj.history(
            period=period,
            interval=interval,
            auto_adjust=False,
        )
        if data is None:
            return pd.DataFrame()
        return data
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def get_info(ticker):
    try:
        return yf.Ticker(ticker.upper()).get_info()
    except Exception:
        return {}


@st.cache_data(ttl=1800, show_spinner=False)
def get_news(ticker):
    try:
        return yf.Ticker(ticker.upper()).get_news(
            count=20,
            tab="all",
        ) or []
    except Exception:
        return []


# ------------------------------------------------------------
# TECHNICAL ENGINE
# ------------------------------------------------------------

def calculate_technical(history):
    df = history.copy()

    if "Close" not in df.columns:
        return df, {}

    close = df["Close"].astype(float)

    for window in [20, 50, 100, 200]:
        df[f"MA{window}"] = close.rolling(window).mean()

    # EMA
    df["EMA12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA26"] = close.ewm(span=26, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()

    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_SIGNAL"] = df["MACD"].ewm(
        span=9,
        adjust=False,
    ).mean()

    df["MACD_HIST"] = (
        df["MACD"] - df["MACD_SIGNAL"]
    )

    # Bollinger Bands
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()

    df["BB_MID"] = bb_mid
    df["BB_UPPER"] = bb_mid + 2 * bb_std
    df["BB_LOWER"] = bb_mid - 2 * bb_std

    # ATR
    high = df["High"]
    low = df["Low"]
    previous_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - previous_close).abs()
    tr3 = (low - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)

    df["ATR14"] = true_range.rolling(14).mean()

    # Momentum
    df["MOM_1M"] = close / close.shift(21) - 1
    df["MOM_3M"] = close / close.shift(63) - 1
    df["MOM_6M"] = close / close.shift(126) - 1

    # Volatility
    daily_returns = close.pct_change()
    df["VOL_20D"] = (
        daily_returns.rolling(20).std()
        * np.sqrt(252)
    )

    # Drawdown
    rolling_high = close.cummax()
    df["DRAWDOWN"] = close / rolling_high - 1

    latest = df.iloc[-1]

    score = 50.0

    checks = [
        ("MA20", 5, -5),
        ("MA50", 7, -6),
        ("MA200", 10, -10),
    ]

    for column, positive, negative in checks:
        value = safe_float(latest.get(column))
        if not np.isnan(value):
            score += (
                positive
                if latest["Close"] > value
                else negative
            )

    rsi = safe_float(latest.get("RSI"))

    if not np.isnan(rsi):
        if 40 <= rsi <= 65:
            score += 8
        elif rsi < 30:
            score += 4
        elif rsi > 75:
            score -= 9

    macd = safe_float(latest.get("MACD"))
    signal = safe_float(latest.get("MACD_SIGNAL"))

    if not np.isnan(macd) and not np.isnan(signal):
        score += 7 if macd > signal else -6

    momentum = safe_float(latest.get("MOM_3M"))

    if not np.isnan(momentum):
        if momentum > 0.15:
            score += 8
        elif momentum > 0:
            score += 4
        elif momentum < -0.15:
            score -= 8

    return df, {
        "price": safe_float(latest["Close"]),
        "ma20": safe_float(latest.get("MA20")),
        "ma50": safe_float(latest.get("MA50")),
        "ma100": safe_float(latest.get("MA100")),
        "ma200": safe_float(latest.get("MA200")),
        "rsi": rsi,
        "macd": macd,
        "signal": signal,
        "atr": safe_float(latest.get("ATR14")),
        "bb_upper": safe_float(latest.get("BB_UPPER")),
        "bb_lower": safe_float(latest.get("BB_LOWER")),
        "momentum_1m": safe_float(latest.get("MOM_1M")),
        "momentum_3m": safe_float(latest.get("MOM_3M")),
        "momentum_6m": safe_float(latest.get("MOM_6M")),
        "volatility": safe_float(latest.get("VOL_20D")),
        "drawdown": safe_float(latest.get("DRAWDOWN")),
        "score": clamp(score),
    }


# ------------------------------------------------------------
# FUNDAMENTAL ENGINE
# ------------------------------------------------------------

def fundamental_analysis(info):
    def v(key):
        return safe_float(info.get(key))

    revenue_growth = v("revenueGrowth")
    earnings_growth = v("earningsGrowth")
    gross_margin = v("grossMargins")
    operating_margin = v("operatingMargins")
    profit_margin = v("profitMargins")
    roe = v("returnOnEquity")
    roa = v("returnOnAssets")

    debt_equity = v("debtToEquity")
    current_ratio = v("currentRatio")

    free_cash_flow = v("freeCashflow")
    operating_cash_flow = v("operatingCashflow")

    pe = v("trailingPE")
    forward_pe = v("forwardPE")
    peg = v("pegRatio")
    price_sales = v("priceToSalesTrailing12Months")
    price_book = v("priceToBook")
    ev_ebitda = v("enterpriseToEbitda")

    # Growth
    growth = 50

    if not np.isnan(revenue_growth):
        growth += np.clip(
            revenue_growth * 80,
            -25,
            25,
        )

    if not np.isnan(earnings_growth):
        growth += np.clip(
            earnings_growth * 50,
            -20,
            20,
        )

    growth_score = clamp(growth)

    # Quality
    quality = 50

    if not np.isnan(gross_margin):
        quality += (
            12 if gross_margin > 0.50
            else 7 if gross_margin > 0.30
            else -5
        )

    if not np.isnan(operating_margin):
        quality += (
            15 if operating_margin > 0.20
            else 8 if operating_margin > 0.10
            else -8
        )

    if not np.isnan(profit_margin):
        quality += (
            12 if profit_margin > 0.20
            else 6 if profit_margin > 0.10
            else -6
        )

    if not np.isnan(roe):
        quality += (
            12 if roe > 0.20
            else 6 if roe > 0.10
            else -6
        )

    quality_score = clamp(quality)

    # Balance sheet
    balance = 60

    if not np.isnan(debt_equity):
        if debt_equity < 50:
            balance += 15
        elif debt_equity > 200:
            balance -= 20

    if not np.isnan(current_ratio):
        if current_ratio >= 1.5:
            balance += 10
        elif current_ratio < 1:
            balance -= 15

    balance_score = clamp(balance)

    # Cash generation
    cash_score = 55

    if not np.isnan(free_cash_flow):
        cash_score = (
            85 if free_cash_flow > 0
            else 30
        )

    # Valuation
    valuation = 55

    if not np.isnan(peg):
        if peg < 1:
            valuation += 20
        elif peg < 1.5:
            valuation += 10
        elif peg > 2.5:
            valuation -= 20

    elif not np.isnan(pe):
        if pe < 18:
            valuation += 15
        elif pe > 40:
            valuation -= 20
        elif pe > 30:
            valuation -= 10

    if (
        not np.isnan(pe)
        and not np.isnan(forward_pe)
    ):
        if forward_pe < pe:
            valuation += 8
        else:
            valuation -= 8

    valuation_score = clamp(valuation)

    total = (
        growth_score * 0.25
        + quality_score * 0.25
        + balance_score * 0.15
        + cash_score * 0.15
        + valuation_score * 0.20
    )

    return {
        "growth_score": growth_score,
        "quality_score": quality_score,
        "balance_score": balance_score,
        "cash_score": cash_score,
        "valuation_score": valuation_score,
        "score": clamp(total),
        "revenue_growth": revenue_growth,
        "earnings_growth": earnings_growth,
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "profit_margin": profit_margin,
        "roe": roe,
        "roa": roa,
        "debt_equity": debt_equity,
        "current_ratio": current_ratio,
        "free_cash_flow": free_cash_flow,
        "operating_cash_flow": operating_cash_flow,
        "pe": pe,
        "forward_pe": forward_pe,
        "peg": peg,
        "price_sales": price_sales,
        "price_book": price_book,
        "ev_ebitda": ev_ebitda,
    }


# ------------------------------------------------------------
# MOAT ENGINE
# ------------------------------------------------------------

def moat_analysis(info):
    def v(key):
        return safe_float(info.get(key))

    score = 50

    margins = v("profitMargins")
    operating_margin = v("operatingMargins")
    roe = v("returnOnEquity")
    growth = v("revenueGrowth")

    if not np.isnan(margins):
        score += (
            15 if margins > 0.20
            else 8 if margins > 0.10
            else -5
        )

    if not np.isnan(operating_margin):
        score += (
            12 if operating_margin > 0.20
            else 6 if operating_margin > 0.10
            else -5
        )

    if not np.isnan(roe):
        score += (
            15 if roe > 0.20
            else 8 if roe > 0.10
            else -5
        )

    if not np.isnan(growth):
        score += (
            15 if growth > 0.15
            else 8 if growth > 0.05
            else -5
        )

    return clamp(score)


# ------------------------------------------------------------
# NEWS ENGINE
# ------------------------------------------------------------

def normalize_news(raw_news):
    output = []

    for item in raw_news:
        content = item.get("content", item)

        title = (
            content.get("title")
            or item.get("title")
            or ""
        )

        title = str(title).strip()

        if not title:
            continue

        lower = title.lower()

        positive = sum(
            word in lower
            for word in POSITIVE_WORDS
        )

        negative = sum(
            word in lower
            for word in NEGATIVE_WORDS
        )

        net = positive - negative

        if net > 0:
            sentiment = "🟢 Bullish"
        elif net < 0:
            sentiment = "🔴 Bearish"
        else:
            sentiment = "🟡 Neutral"

        publisher = "Yahoo Finance"

        provider = content.get("provider")

        if isinstance(provider, dict):
            publisher = (
                provider.get("displayName")
                or publisher
            )

        url = ""

        for key in [
            "clickThroughUrl",
            "canonicalUrl",
        ]:
            obj = content.get(key)

            if isinstance(obj, dict):
                url = obj.get("url", "")

                if url:
                    break

        output.append(
            {
                "title": title,
                "publisher": publisher,
                "sentiment": sentiment,
                "url": url,
                "score": net,
            }
        )

    return output


def news_analysis(raw_news):
    rows = normalize_news(raw_news)

    if not rows:
        return rows, 50

    scores = [
        row["score"]
        for row in rows
    ]

    average = np.mean(scores)

    score = clamp(
        50 + average * 15
    )

    return rows, score


# ------------------------------------------------------------
# RISK ENGINE
# ------------------------------------------------------------

def risk_analysis(info, technical):
    beta = safe_float(info.get("beta"))

    risk = 40

    if not np.isnan(beta):
        if beta > 1.7:
            risk += 25
        elif beta > 1.3:
            risk += 15
        elif beta < 0.8:
            risk -= 10

    volatility = technical.get("volatility")

    if not np.isnan(volatility):
        if volatility > 0.60:
            risk += 20
        elif volatility > 0.40:
            risk += 10
        elif volatility < 0.20:
            risk -= 8

    return clamp(risk)


# ------------------------------------------------------------
# ANTI-HYPE ENGINE
# ------------------------------------------------------------

def anti_hype_analysis(technical, news_score):
    hype = 0

    rsi = technical.get("rsi")
    momentum = technical.get("momentum_1m")
    volatility = technical.get("volatility")

    if not np.isnan(rsi):
        if rsi > 75:
            hype += 30
        elif rsi > 70:
            hype += 20

    if not np.isnan(momentum):
        if momentum > 0.25:
            hype += 30
        elif momentum > 0.15:
            hype += 15

    if news_score > 75:
        hype += 20

    if not np.isnan(volatility):
        if volatility > 0.60:
            hype += 20

    hype = clamp(hype)

    if hype >= 70:
        level = "🔴 HIGH"
    elif hype >= 40:
        level = "🟡 MEDIUM"
    else:
        level = "🟢 LOW"

    return hype, level


# ------------------------------------------------------------
# FIRST PRINCIPLES DATA
# ------------------------------------------------------------

def first_principles(info, fundamental):
    company = (
        info.get("longName")
        or info.get("shortName")
        or "Unknown"
    )

    sector = (
        info.get("sector")
        or "Unknown"
    )

    industry = (
        info.get("industry")
        or "Unknown"
    )

    business = (
        info.get("longBusinessSummary")
        or "暂无公司业务简介。"
    )

    return {
        "company": company,
        "sector": sector,
        "industry": industry,
        "business": business,
        "growth": fundamental["growth_score"],
        "quality": fundamental["quality_score"],
        "valuation": fundamental["valuation_score"],
    }


# ------------------------------------------------------------
# MASTER ENGINE
# ------------------------------------------------------------

def analyze_stock(ticker, period="1y"):
    ticker = ticker.upper().strip()

    data = get_history(
        ticker,
        period=period,
    )

    if data.empty:
        return None

    info = get_info(ticker)

    technical_df, technical = calculate_technical(
        data
    )

    fundamental = fundamental_analysis(
        info
    )

    moat = moat_analysis(info)

    raw_news = get_news(ticker)

    news_rows, news_score = news_analysis(
        raw_news
    )

    risk = risk_analysis(
        info,
        technical,
    )

    hype_score, hype_level = anti_hype_analysis(
        technical,
        news_score,
    )

    # Main Simon Score
    score = (
        technical["score"] * 0.25
        + fundamental["score"] * 0.30
        + fundamental["valuation_score"] * 0.10
        + moat * 0.15
        + news_score * 0.08
        + (100 - risk) * 0.07
        + (100 - hype_score) * 0.05
    )

    score = clamp(score)

    # Risk / reward proxy
    upside_quality = (
        fundamental["growth_score"] * 0.4
        + moat * 0.3
        + technical["score"] * 0.3
    )

    downside_risk = (
        risk * 0.6
        + hype_score * 0.4
    )

    risk_reward = clamp(
        50
        + upside_quality * 0.45
        - downside_risk * 0.35
    )

    return {
        "ticker": ticker,
        "data": technical_df,
        "info": info,
        "technical": technical,
        "fundamental": fundamental,
        "moat": moat,
        "news": news_rows,
        "news_score": news_score,
        "risk": risk,
        "hype_score": hype_score,
        "hype_level": hype_level,
        "risk_reward": risk_reward,
        "score": score,
        "grade": grade(score),
        "action": action(score),
        "first_principles": first_principles(
            info,
            fundamental,
        ),
    }


# ------------------------------------------------------------
# AI ENGINE
# ------------------------------------------------------------

def ai_available():
    key = (
        st.secrets.get(
            "OPENAI_API_KEY",
            os.getenv("OPENAI_API_KEY", ""),
        )
    )

    return bool(key)


def run_ai(prompt):
    if not ai_available():
        return (
            "⚠️ **AI 尚未配置**\n\n"
            "Simon Quant Engine 可以正常运行。"
            "如果要开启 AI Research，请在 Streamlit "
            "Secrets 中加入 `OPENAI_API_KEY`。"
        )

    try:
        from openai import OpenAI

        key = st.secrets.get(
            "OPENAI_API_KEY",
            os.getenv("OPENAI_API_KEY"),
        )

        model = st.secrets.get(
            "SIMON_AI_MODEL",
            os.getenv(
                "SIMON_AI_MODEL",
                "gpt-5.6-luna",
            ),
        )

        client = OpenAI(api_key=key)

        response = client.responses.create(
            model=model,
            instructions=(
                "You are Simon Stock AI, an advanced "
                "US equity research assistant. "
                "Use supplied facts only. "
                "Never invent prices or financial data. "
                "Clearly separate facts, assumptions, "
                "and interpretation. "
                "Do not guarantee returns. "
                "Think from first principles."
            ),
            input=prompt,
        )

        return response.output_text

    except Exception as exc:
        return (
            f"⚠️ AI 调用失败：{exc}\n\n"
            "Quant Engine 仍然可以正常使用。"
        )


def build_ai_prompt(result):
    info = result["info"]
    technical = result["technical"]
    fundamental = result["fundamental"]

    news_titles = [
        row["title"]
        for row in result["news"][:12]
    ]

    payload = {
        "ticker": result["ticker"],
        "company": info.get("longName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "price": technical["price"],
        "simon_score": result["score"],
        "grade": result["grade"],
        "action": result["action"],
        "risk_reward": result["risk_reward"],
        "risk": result["risk"],
        "hype": result["hype_level"],
        "moat": result["moat"],
        "technical": technical,
        "fundamental": fundamental,
        "news": news_titles,
    }

    return """
请对下面股票进行 Simon Stock V11 AI Ultimate 分析。

必须使用 First Principles 思维。

请按照以下结构回答：

1. Executive Summary
2. First Principles
   - 公司怎么赚钱
   - 核心增长来源
   - 最大竞争优势
   - 最容易被破坏的假设
3. Technical Analysis
4. Fundamental Analysis
5. Valuation
6. Moat
7. Daily News / Catalysts
8. Anti-Hype Risk
9. Bull Case
10. Bear Case
11. Base Case
12. Risk / Reward
13. AI Judge
14. What would make us change our mind?
15. Simon Verdict

AI Judge 必须真正比较 Bull 和 Bear 的证据，而不是简单折中。

如果数据不足，明确说数据不足。

不要保证收益。

不要虚构目标价。

以下是程序计算的数据：

""" + json.dumps(
        payload,
        ensure_ascii=False,
        default=str,
        indent=2,
    )


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Simon Control")

    ticker = st.text_input(
        "🔎 股票代码",
        value="AAPL",
    ).strip().upper()

    period = st.selectbox(
        "📅 数据周期",
        [
            "6mo",
            "1y",
            "2y",
            "5y",
        ],
        index=1,
    )

    st.divider()

    st.subheader("⭐ Watchlist")

    new_symbol = st.text_input(
        "添加股票",
        placeholder="AMD",
    )

    if st.button(
        "➕ Add",
        use_container_width=True,
    ):
        symbol = new_symbol.strip().upper()

        if (
            symbol
            and symbol not in st.session_state.watchlist
        ):
            st.session_state.watchlist.append(
                symbol
            )
            st.rerun()

    st.write(
        " · ".join(
            st.session_state.watchlist
        )
    )

    st.divider()

    st.subheader("💼 Portfolio")

    portfolio_ticker = st.text_input(
        "股票",
        value="AAPL",
    ).upper()

    portfolio_shares = st.number_input(
        "股数",
        min_value=0.0,
        value=1.0,
        step=1.0,
    )

    portfolio_cost = st.number_input(
        "平均成本",
        min_value=0.0,
        value=0.0,
        step=1.0,
    )

    if st.button(
        "💾 保存持仓",
        use_container_width=True,
    ):
        st.session_state.portfolio.append(
            {
                "ticker": portfolio_ticker,
                "shares": portfolio_shares,
                "cost": portfolio_cost,
            }
        )

        st.success("已加入组合。")


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

st.title("📈 SIMON STOCK V11 AI ULTIMATE")

st.caption(
    "First Principles · Quant · Fundamental · "
    "Valuation · News · Anti-Hype · Portfolio · AI Judge"
)

st.info(
    "💡 V11 的核心思想："
    "不是预测股票明天涨跌，而是判断当前价格背后的假设是否合理。"
)


# ------------------------------------------------------------
# MARKET DASHBOARD
# ------------------------------------------------------------

st.header("🌎 Market Intelligence")

market_cols = st.columns(5)

for col, (name, symbol) in zip(
    market_cols,
    MARKET_SYMBOLS.items(),
):
    market_history = get_history(
        symbol,
        period="5d",
    )

    if not market_history.empty:
        current = safe_float(
            market_history["Close"].iloc[-1]
        )

        if len(market_history) >= 2:
            previous = safe_float(
                market_history["Close"].iloc[-2]
            )
        else:
            previous = current

        change = (
            (current / previous - 1) * 100
            if previous
            else 0
        )

        col.metric(
            name,
            f"{current:,.2f}",
            f"{change:.2f}%",
        )


# ------------------------------------------------------------
# MASTER STOCK ANALYSIS
# ------------------------------------------------------------

result = analyze_stock(
    ticker,
    period,
)

if result is None:
    st.error(
        f"无法获取 {ticker} 数据。"
        "请检查股票代码或稍后重试。"
    )

    st.stop()


# ------------------------------------------------------------
# SCORE HEADER
# ------------------------------------------------------------

st.header(
    f"🔬 {ticker} — "
    f"{result['info'].get('longName', '')}"
)

score_cols = st.columns(5)

score_cols[0].metric(
    "Price",
    fmt_money(
        result["technical"]["price"]
    ),
)

score_cols[1].metric(
    "Simon Score",
    f"{result['score']:.0f}/100",
)

score_cols[2].metric(
    "Grade",
    result["grade"],
)

score_cols[3].metric(
    "Action",
    result["action"],
)

score_cols[4].metric(
    "Risk/Reward",
    f"{result['risk_reward']:.0f}/100",
)


# ------------------------------------------------------------
# SCORE BREAKDOWN
# ------------------------------------------------------------

st.subheader("🧮 Simon Score Breakdown")

breakdown = pd.DataFrame(
    {
        "Engine": [
            "Technical",
            "Fundamental",
            "Valuation",
            "Moat",
            "News",
            "Risk",
            "Anti-Hype",
        ],
        "Score": [
            result["technical"]["score"],
            result["fundamental"]["score"],
            result["fundamental"]["valuation_score"],
            result["moat"],
            result["news_score"],
            100 - result["risk"],
            100 - result["hype_score"],
        ],
    }
)

st.dataframe(
    breakdown,
    use_container_width=True,
    hide_index=True,
)

st.bar_chart(
    breakdown.set_index("Engine")
)


# ------------------------------------------------------------
# TECHNICAL
# ------------------------------------------------------------

st.header("📊 Technical Engine")

chart_columns = [
    "Close",
    "MA20",
    "MA50",
    "MA200",
]

existing_columns = [
    column
    for column in chart_columns
    if column in result["data"].columns
]

st.line_chart(
    result["data"][existing_columns]
)

tech_cols = st.columns(6)

tech_cols[0].metric(
    "RSI",
    (
        f"{result['technical']['rsi']:.1f}"
        if not np.isnan(
            result["technical"]["rsi"]
        )
        else "—"
    ),
)

tech_cols[1].metric(
    "MA20",
    fmt_money(
        result["technical"]["ma20"]
    ),
)

tech_cols[2].metric(
    "MA50",
    fmt_money(
        result["technical"]["ma50"]
    ),
)

tech_cols[3].metric(
    "MA200",
    fmt_money(
        result["technical"]["ma200"]
    ),
)

tech_cols[4].metric(
    "1M Momentum",
    fmt_pct(
        result["technical"]["momentum_1m"]
    ),
)

tech_cols[5].metric(
    "3M Momentum",
    fmt_pct(
        result["technical"]["momentum_3m"]
    ),
)


# ------------------------------------------------------------
# FUNDAMENTAL
# ------------------------------------------------------------

st.header("🏢 Fundamental Engine")

f = result["fundamental"]

fund_cols = st.columns(6)

fund_cols[0].metric(
    "Growth",
    f"{f['growth_score']:.0f}",
)

fund_cols[1].metric(
    "Quality",
    f"{f['quality_score']:.0f}",
)

fund_cols[2].metric(
    "Balance",
    f"{f['balance_score']:.0f}",
)

fund_cols[3].metric(
    "Cash Flow",
    f"{f['cash_score']:.0f}",
)

fund_cols[4].metric(
    "Valuation",
    f"{f['valuation_score']:.0f}",
)

fund_cols[5].metric(
    "Moat",
    f"{result['moat']:.0f}",
)

fundamental_table = pd.DataFrame(
    {
        "Metric": [
            "Revenue Growth",
            "EPS Growth",
            "Gross Margin",
            "Operating Margin",
            "Profit Margin",
            "ROE",
            "ROA",
            "Debt / Equity",
            "Current Ratio",
            "P/E",
            "Forward P/E",
            "PEG",
            "P/S",
            "P/B",
            "EV/EBITDA",
            "Free Cash Flow",
        ],
        "Value": [
            fmt_pct(f["revenue_growth"]),
            fmt_pct(f["earnings_growth"]),
            fmt_pct(f["gross_margin"]),
            fmt_pct(f["operating_margin"]),
            fmt_pct(f["profit_margin"]),
            fmt_pct(f["roe"]),
            fmt_pct(f["roa"]),
            (
                "—"
                if np.isnan(f["debt_equity"])
                else f"{f['debt_equity']:.1f}"
            ),
            (
                "—"
                if np.isnan(f["current_ratio"])
                else f"{f['current_ratio']:.2f}"
            ),
            (
                "—"
                if np.isnan(f["pe"])
                else f"{f['pe']:.2f}"
            ),
            (
                "—"
                if np.isnan(f["forward_pe"])
                else f"{f['forward_pe']:.2f}"
            ),
            (
                "—"
                if np.isnan(f["peg"])
                else f"{f['peg']:.2f}"
            ),
            (
                "—"
                if np.isnan(f["price_sales"])
                else f"{f['price_sales']:.2f}"
            ),
            (
                "—"
                if np.isnan(f["price_book"])
                else f"{f['price_book']:.2f}"
            ),
            (
                "—"
                if np.isnan(f["ev_ebitda"])
                else f"{f['ev_ebitda']:.2f}"
            ),
            fmt_money(f["free_cash_flow"]),
        ],
    }
)

st.dataframe(
    fundamental_table,
    use_container_width=True,
    hide_index=True,
)


# ------------------------------------------------------------
# FIRST PRINCIPLES
# ------------------------------------------------------------

st.header("🧠 First Principles")

fp = result["first_principles"]

st.markdown(
    f"""
### {fp['company']}

**Sector:** {fp['sector']}  
**Industry:** {fp['industry']}

**Business**

{fp['business']}
"""
)

st.markdown(
    """
### 核心问题

**① 公司到底靠什么赚钱？**

**② 增长来自什么？**

**③ 竞争优势能持续多久？**

**④ 当前股价隐含了多高的增长预期？**

**⑤ 什么情况会让整个投资逻辑失效？**
"""
)


# ------------------------------------------------------------
# NEWS
# ------------------------------------------------------------

st.header("📰 Daily Stock News")

st.metric(
    "News Sentiment Score",
    f"{result['news_score']:.0f}/100",
)

if result["news"]:
    for item in result["news"][:12]:
        st.markdown(
            f"### {item['sentiment']} "
            f"{item['title']}"
        )

        st.caption(
            item["publisher"]
        )

        if item["url"]:
            st.link_button(
                "阅读原文",
                item["url"],
            )
else:
    st.info(
        "暂时没有抓到新闻。"
    )


# ------------------------------------------------------------
# ANTI-HYPE
# ------------------------------------------------------------

st.header("🚨 Anti-Hype Engine")

a, b, c = st.columns(3)

a.metric(
    "Hype Score",
    f"{result['hype_score']:.0f}/100",
)

b.metric(
    "Hype Risk",
    result["hype_level"],
)

c.metric(
    "Risk Score",
    f"{result['risk']:.0f}/100",
)

if result["hype_score"] >= 70:
    st.warning(
        "⚠️ 当前可能存在明显追热点风险："
        "高动量、超买、新闻热度或波动率可能同时出现。"
    )
elif result["hype_score"] >= 40:
    st.info(
        "🟡 当前存在一定热度风险，"
        "不建议仅凭新闻追涨。"
    )
else:
    st.success(
        "🟢 暂未检测到明显过热信号。"
    )


# ------------------------------------------------------------
# WATCHLIST SCANNER
# ------------------------------------------------------------

st.header("⭐ Watchlist Scanner")

scan_button = st.button(
    "🚀 扫描全部 Watchlist",
    type="primary",
)

if scan_button:
    rows = []

    progress = st.progress(0)

    total = len(
        st.session_state.watchlist
    )

    for index, symbol in enumerate(
        st.session_state.watchlist
    ):
        stock = analyze_stock(
            symbol,
            "6mo",
        )

        if stock:
            rows.append(
                {
                    "Ticker": symbol,
                    "Price": stock["technical"]["price"],
                    "Simon Score": round(
                        stock["score"],
                        1,
                    ),
                    "Grade": stock["grade"],
                    "Action": stock["action"],
                    "Moat": round(
                        stock["moat"],
                        1,
                    ),
                    "Risk": round(
                        stock["risk"],
                        1,
                    ),
                    "Hype": round(
                        stock["hype_score"],
                        1,
                    ),
                }
            )

        progress.progress(
            (index + 1) / total
        )

    if rows:
        scanner_df = pd.DataFrame(
            rows
        ).sort_values(
            "Simon Score",
            ascending=False,
        )

        st.dataframe(
            scanner_df,
            use_container_width=True,
            hide_index=True,
        )

        st.bar_chart(
            scanner_df.set_index(
                "Ticker"
            )["Simon Score"]
        )


# ------------------------------------------------------------
# PORTFOLIO
# ------------------------------------------------------------

st.header("💼 Portfolio Intelligence")

if not st.session_state.portfolio:
    st.info(
        "还没有持仓。"
        "可以在左侧添加股票。"
    )
else:
    portfolio_rows = []

    total_value = 0
    total_cost = 0

    for position in (
        st.session_state.portfolio
    ):
        stock = analyze_stock(
            position["ticker"],
            "6mo",
        )

        if not stock:
            continue

        price = stock["technical"]["price"]

        shares = position["shares"]
        cost = position["cost"]

        value = price * shares
        cost_total = cost * shares

        total_value += value
        total_cost += cost_total

        pnl = value - cost_total

        portfolio_rows.append(
            {
                "Ticker": position["ticker"],
                "Shares": shares,
                "Avg Cost": cost,
                "Price": price,
                "Value": value,
                "P/L": pnl,
                "P/L %": (
                    pnl / cost_total * 100
                    if cost_total
                    else np.nan
                ),
                "Simon Score": stock["score"],
            }
        )

    if portfolio_rows:
        portfolio_df = pd.DataFrame(
            portfolio_rows
        )

        st.dataframe(
            portfolio_df,
            use_container_width=True,
            hide_index=True,
        )

        p1, p2, p3, p4 = st.columns(4)

        p1.metric(
            "Portfolio Value",
            fmt_money(total_value),
        )

        p2.metric(
            "Total Cost",
            fmt_money(total_cost),
        )

        p3.metric(
            "P/L",
            fmt_money(
                total_value - total_cost
            ),
        )

        p4.metric(
            "P/L %",
            (
                f"{(total_value / total_cost - 1) * 100:.2f}%"
                if total_cost
                else "—"
            ),
        )


# ------------------------------------------------------------
# AI RESEARCH
# ------------------------------------------------------------

st.header("🤖 Simon AI Research")

if not ai_available():
    st.info(
        "AI API 未配置，但 V11 Quant / News / "
        "Fundamental / First Principles 模块仍然可以使用。"
    )

if st.button(
    "🧠 Run Bull vs Bear + AI Judge",
    type="primary",
):
    prompt = build_ai_prompt(
        result
    )

    with st.spinner(
        "Simon AI 正在进行 First Principles + Bull/Bear Debate..."
    ):
        ai_report = run_ai(
            prompt
        )

    st.session_state.last_ai_report = (
        ai_report
    )

if st.session_state.last_ai_report:
    st.markdown(
        st.session_state.last_ai_report
    )


# ------------------------------------------------------------
# DAILY MARKET AI
# ------------------------------------------------------------

st.header("🌅 Simon Daily Market Report")

if st.button(
    "📋 Generate Daily Market Report"
):
    market_snapshot = {}

    for name, symbol in (
        MARKET_SYMBOLS.items()
    ):
        h = get_history(
            symbol,
            "5d",
        )

        if not h.empty:
            current = safe_float(
                h["Close"].iloc[-1]
            )

            previous = (
                safe_float(
                    h["Close"].iloc[-2]
                )
                if len(h) >= 2
                else current
            )

            market_snapshot[name] = {
                "price": current,
                "change_pct": (
                    current / previous - 1
                )
                if previous
                else 0,
            }

    prompt = """
请生成一份 Simon Stock Daily Market Report。

必须包括：

1. 今日市场状态
2. Risk-on / Risk-off 判断
3. Nasdaq / S&P 500 / Dow / Russell / VIX
4. 科技股与 AI 板块风险
5. 今日最重要的市场变量
6. 今日应该关注什么
7. 今日最不应该追什么
8. Bull Case
9. Bear Case
10. 最终市场结论

不要编造数据。

当前数据：

""" + json.dumps(
        market_snapshot,
        ensure_ascii=False,
        default=str,
        indent=2,
    )

    with st.spinner(
        "AI 正在生成 Daily Market Report..."
    ):
        report = run_ai(
            prompt
        )

    st.markdown(report)


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.divider()

st.caption(
    "Simon Stock V11 AI Ultimate · "
    "Quant + Fundamental + News + AI Research"
)

st.caption(
    "⚠️ This application is for research and "
    "educational purposes only. It is not financial advice."
)