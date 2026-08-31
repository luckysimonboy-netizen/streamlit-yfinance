import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ============================================================
# SIMON STOCK V9
# AI-Native US Stock Research Terminal
#
# Data: Yahoo Finance via yfinance
# UI: Streamlit
# No OpenAI API required
#
# Modules:
#   1. Dashboard
#   2. Simon Score 3.0
#   3. Technical Analysis
#   4. Fundamental Analysis
#   5. Valuation
#   6. Bull / Base / Bear
#   7. Risk Radar
#   8. Watchlist Scanner
#   9. Stock Battle
#  10. Portfolio
#  11. Research Report
# ============================================================


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Simon Stock V9",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        color: #777;
        margin-bottom: 1.5rem;
    }

    .card {
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,.25);
        margin-bottom: 1rem;
    }

    .big-score {
        font-size: 3rem;
        font-weight: 800;
    }

    .small-text {
        color: #777;
        font-size: .85rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "watchlist" not in st.session_state:
    st.session_state.watchlist = [
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

if "portfolio" not in st.session_state:
    st.session_state.portfolio = []


# ============================================================
# HELPERS
# ============================================================

def safe_num(value):
    try:
        value = float(value)

        if np.isfinite(value):
            return value

        return np.nan

    except Exception:
        return np.nan


def safe_div(a, b):
    a = safe_num(a)
    b = safe_num(b)

    if not np.isfinite(a):
        return np.nan

    if not np.isfinite(b) or b == 0:
        return np.nan

    return a / b


def money(value):
    value = safe_num(value)

    if not np.isfinite(value):
        return "—"

    return f"${value:,.2f}"


def percent(value):
    value = safe_num(value)

    if not np.isfinite(value):
        return "—"

    return f"{value * 100:.1f}%"


def number(value):
    value = safe_num(value)

    if not np.isfinite(value):
        return "—"

    return f"{value:,.2f}"


def compact_number(value):
    value = safe_num(value)

    if not np.isfinite(value):
        return "—"

    if abs(value) >= 1e12:
        return f"{value / 1e12:.2f}T"

    if abs(value) >= 1e9:
        return f"{value / 1e9:.2f}B"

    if abs(value) >= 1e6:
        return f"{value / 1e6:.2f}M"

    if abs(value) >= 1e3:
        return f"{value / 1e3:.2f}K"

    return f"{value:.2f}"


def info_value(info, *keys):

    for key in keys:

        value = safe_num(info.get(key))

        if np.isfinite(value):
            return value

    return np.nan


def score_label(score):

    if score >= 90:
        return "S级"

    if score >= 80:
        return "A级"

    if score >= 70:
        return "B级"

    if score >= 60:
        return "C级"

    if score >= 50:
        return "D级"

    return "E级"


def score_action(score):

    if score >= 90:
        return "🔥 强烈关注"

    if score >= 80:
        return "🟢 分批买入"

    if score >= 70:
        return "🟢 逢回调买"

    if score >= 60:
        return "🔵 持有 / 观察"

    if score >= 50:
        return "🟡 等待更好价格"

    if score >= 40:
        return "🟠 谨慎"

    return "🔴 减仓 / 回避"


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def get_history(ticker, period):

    try:

        data = yf.Ticker(ticker).history(
            period=period,
            auto_adjust=False
        )

        return data

    except Exception:

        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def get_company_data(ticker):

    ticker_obj = yf.Ticker(ticker)

    try:
        info = ticker_obj.info
    except Exception:
        info = {}

    try:
        financials = ticker_obj.financials
    except Exception:
        financials = pd.DataFrame()

    try:
        balance = ticker_obj.balance_sheet
    except Exception:
        balance = pd.DataFrame()

    try:
        cashflow = ticker_obj.cashflow
    except Exception:
        cashflow = pd.DataFrame()

    return info, financials, balance, cashflow


# ============================================================
# FINANCIAL DATA EXTRACTION
# ============================================================

def dataframe_value(df, names):

    if df is None or df.empty:
        return np.nan

    for name in names:

        if name in df.index:

            try:

                series = pd.to_numeric(
                    df.loc[name],
                    errors="coerce"
                ).dropna()

                if len(series) > 0:
                    return float(series.iloc[0])

            except Exception:
                pass

    return np.nan


# ============================================================
# TECHNICAL ENGINE
# ============================================================

def technical_analysis(history):

    df = history.copy()

    close = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    volume = pd.to_numeric(
        df["Volume"],
        errors="coerce"
    )

    # Moving averages

    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()
    df["MA200"] = close.rolling(200).mean()

    # RSI

    delta = close.diff()

    gain = delta.clip(lower=0).rolling(14).mean()

    loss = (
        -delta.clip(upper=0)
        .rolling(14)
        .mean()
    )

    rs = gain / loss.replace(0, np.nan)

    df["RSI"] = 100 - (
        100 / (1 + rs)
    )

    # MACD

    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_SIGNAL"] = (
        df["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    # Volatility

    df["VOLATILITY"] = (
        close.pct_change()
        .rolling(20)
        .std()
        * np.sqrt(252)
    )

    # Momentum

    df["RETURN_1M"] = (
        close / close.shift(21) - 1
    )

    df["RETURN_3M"] = (
        close / close.shift(63) - 1
    )

    df["RETURN_6M"] = (
        close / close.shift(126) - 1
    )

    # Volume ratio

    df["AVG_VOLUME20"] = volume.rolling(20).mean()

    df["VOLUME_RATIO"] = (
        volume / df["AVG_VOLUME20"]
    )

    last = df.iloc[-1]

    price = safe_num(last["Close"])
    ma20 = safe_num(last["MA20"])
    ma50 = safe_num(last["MA50"])
    ma200 = safe_num(last["MA200"])
    rsi = safe_num(last["RSI"])
    macd = safe_num(last["MACD"])
    macd_signal = safe_num(
        last["MACD_SIGNAL"]
    )
    volatility = safe_num(
        last["VOLATILITY"]
    )

    score = 50

    reasons = []

    # MA20

    if np.isfinite(price) and np.isfinite(ma20):

        if price > ma20:

            score += 5
            reasons.append(
                "价格站上 MA20"
            )

        else:

            score -= 4
            reasons.append(
                "价格低于 MA20"
            )

    # MA50

    if np.isfinite(price) and np.isfinite(ma50):

        if price > ma50:

            score += 7
            reasons.append(
                "价格站上 MA50"
            )

        else:

            score -= 6
            reasons.append(
                "价格低于 MA50"
            )

    # MA200

    if np.isfinite(price) and np.isfinite(ma200):

        if price > ma200:

            score += 9
            reasons.append(
                "价格站上 MA200"
            )

        else:

            score -= 9
            reasons.append(
                "价格低于 MA200"
            )

    # MA trend

    if np.isfinite(ma50) and np.isfinite(ma200):

        if ma50 > ma200:

            score += 7
            reasons.append(
                "MA50 > MA200，中长期趋势偏多"
            )

        else:

            score -= 6
            reasons.append(
                "MA50 < MA200，中长期趋势偏弱"
            )

    # RSI

    if np.isfinite(rsi):

        if 45 <= rsi <= 65:

            score += 7
            reasons.append(
                "RSI处于相对健康区域"
            )

        elif rsi < 30:

            score += 6
            reasons.append(
                "RSI进入超卖区域"
            )

        elif rsi > 75:

            score -= 8
            reasons.append(
                "RSI偏热"
            )

        elif rsi > 65:

            score -= 2
            reasons.append(
                "RSI偏强"
            )

    # MACD

    if (
        np.isfinite(macd)
        and np.isfinite(macd_signal)
    ):

        if macd > macd_signal:

            score += 7
            reasons.append(
                "MACD偏多"
            )

        else:

            score -= 6
            reasons.append(
                "MACD偏空"
            )

    # Volatility

    if np.isfinite(volatility):

        if volatility > 0.65:

            score -= 7
            reasons.append(
                "年化波动率较高"
            )

        elif volatility < 0.25:

            score += 3
            reasons.append(
                "波动率相对温和"
            )

    score = float(
        np.clip(score, 0, 100)
    )

    return {
        "price": price,
        "ma20": ma20,
        "ma50": ma50,
        "ma200": ma200,
        "rsi": rsi,
        "macd": macd,
        "macd_signal": macd_signal,
        "volatility": volatility,
        "score": score,
        "reasons": reasons,
    }, df


# ============================================================
# FUNDAMENTAL ENGINE
# ============================================================

def fundamental_analysis(
    info,
    financials,
    balance,
    cashflow
):

    revenue = info_value(
        info,
        "totalRevenue"
    )

    revenue_growth = info_value(
        info,
        "revenueGrowth"
    )

    earnings_growth = info_value(
        info,
        "earningsGrowth"
    )

    gross_margin = info_value(
        info,
        "grossMargins"
    )

    operating_margin = info_value(
        info,
        "operatingMargins"
    )

    profit_margin = info_value(
        info,
        "profitMargins"
    )

    roe = info_value(
        info,
        "returnOnEquity"
    )

    roa = info_value(
        info,
        "returnOnAssets"
    )

    debt_equity = info_value(
        info,
        "debtToEquity"
    )

    current_ratio = info_value(
        info,
        "currentRatio"
    )

    eps = info_value(
        info,
        "trailingEps"
    )

    forward_eps = info_value(
        info,
        "forwardEps"
    )

    pe = info_value(
        info,
        "trailingPE"
    )

    forward_pe = info_value(
        info,
        "forwardPE"
    )

    peg = info_value(
        info,
        "pegRatio"
    )

    ps = info_value(
        info,
        "priceToSalesTrailing12Months"
    )

    pb = info_value(
        info,
        "priceToBook"
    )

    ev_ebitda = info_value(
        info,
        "enterpriseToEbitda"
    )

    beta = info_value(
        info,
        "beta"
    )

    dividend_yield = info_value(
        info,
        "dividendYield"
    )

    if np.isfinite(dividend_yield):

        if dividend_yield > 1:
            dividend_yield /= 100

    operating_cashflow = info_value(
        info,
        "operatingCashflow"
    )

    capital_expenditure = info_value(
        info,
        "capitalExpenditures"
    )

    # Fallback financial statements

    if not np.isfinite(revenue):

        revenue = dataframe_value(
            financials,
            [
                "Total Revenue",
                "Operating Revenue"
            ]
        )

    if not np.isfinite(
        operating_cashflow
    ):

        operating_cashflow = dataframe_value(
            cashflow,
            [
                "Operating Cash Flow",
                "Total Cash From Operating Activities"
            ]
        )

    if not np.isfinite(
        capital_expenditure
    ):

        capital_expenditure = dataframe_value(
            cashflow,
            [
                "Capital Expenditure",
                "Capital Expenditures"
            ]
        )

    # FCF

    free_cash_flow = np.nan

    if np.isfinite(
        operating_cashflow
    ):

        if np.isfinite(
            capital_expenditure
        ):

            if capital_expenditure < 0:

                free_cash_flow = (
                    operating_cashflow
                    + capital_expenditure
                )

            else:

                free_cash_flow = (
                    operating_cashflow
                    - capital_expenditure
                )

        else:

            free_cash_flow = (
                operating_cashflow
            )

    # --------------------------------------------------------
    # Growth score
    # --------------------------------------------------------

    growth_score = 50

    if np.isfinite(
        revenue_growth
    ):

        growth_score += np.clip(
            revenue_growth * 80,
            -25,
            25
        )

    if np.isfinite(
        earnings_growth
    ):

        growth_score += np.clip(
            earnings_growth * 60,
            -25,
            25
        )

    growth_score = np.clip(
        growth_score,
        0,
        100
    )

    # --------------------------------------------------------
    # Quality score
    # --------------------------------------------------------

    quality_score = 50

    if np.isfinite(
        gross_margin
    ):

        quality_score += np.clip(
            (gross_margin - 0.30) * 50,
            -10,
            10
        )

    if np.isfinite(
        operating_margin
    ):

        quality_score += np.clip(
            (operating_margin - 0.10) * 50,
            -10,
            10
        )

    if np.isfinite(roe):

        quality_score += np.clip(
            (roe - 0.12) * 40,
            -12,
            12
        )

    quality_score = np.clip(
        quality_score,
        0,
        100
    )

    # --------------------------------------------------------
    # Balance sheet
    # --------------------------------------------------------

    balance_score = 60

    if np.isfinite(
        debt_equity
    ):

        if debt_equity < 50:

            balance_score += 12

        elif debt_equity > 150:

            balance_score -= 18

    if np.isfinite(
        current_ratio
    ):

        if current_ratio >= 1.5:

            balance_score += 8

        elif current_ratio < 1:

            balance_score -= 12

    balance_score = np.clip(
        balance_score,
        0,
        100
    )

    # --------------------------------------------------------
    # Cash flow
    # --------------------------------------------------------

    cashflow_score = 50

    if np.isfinite(
        free_cash_flow
    ):

        if free_cash_flow > 0:

            cashflow_score += 25

        else:

            cashflow_score -= 25

    cashflow_score = np.clip(
        cashflow_score,
        0,
        100
    )

    # --------------------------------------------------------
    # Valuation
    # --------------------------------------------------------

    valuation_score = 55

    if np.isfinite(peg):

        if peg < 1:

            valuation_score += 18

        elif peg < 1.5:

            valuation_score += 10

        elif peg > 2.5:

            valuation_score -= 15

    elif np.isfinite(pe):

        if pe < 18:

            valuation_score += 15

        elif pe > 35:

            valuation_score -= 15

    if (
        np.isfinite(pe)
        and np.isfinite(forward_pe)
    ):

        if forward_pe < pe * 0.9:

            valuation_score += 8

        elif forward_pe > pe * 1.1:

            valuation_score -= 8

    valuation_score = np.clip(
        valuation_score,
        0,
        100
    )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    overall = (

        growth_score * 0.25
        + quality_score * 0.25
        + balance_score * 0.15
        + cashflow_score * 0.15
        + valuation_score * 0.20

    )

    return {

        "revenue": revenue,

        "revenue_growth": revenue_growth,

        "earnings_growth": earnings_growth,

        "gross_margin": gross_margin,

        "operating_margin": operating_margin,

        "profit_margin": profit_margin,

        "roe": roe,

        "roa": roa,

        "debt_equity": debt_equity,

        "current_ratio": current_ratio,

        "eps": eps,

        "forward_eps": forward_eps,

        "pe": pe,

        "forward_pe": forward_pe,

        "peg": peg,

        "ps": ps,

        "pb": pb,

        "ev_ebitda": ev_ebitda,

        "beta": beta,

        "dividend_yield": dividend_yield,

        "operating_cashflow": operating_cashflow,

        "capital_expenditure": capital_expenditure,

        "free_cash_flow": free_cash_flow,

        "growth_score": growth_score,

        "quality_score": quality_score,

        "balance_score": balance_score,

        "cashflow_score": cashflow_score,

        "valuation_score": valuation_score,

        "score": overall,

    }


# ============================================================
# DCF
# ============================================================

def dcf_per_share(
    fcf,
    shares,
    growth,
    discount_rate,
    terminal_growth
):

    fcf = safe_num(fcf)
    shares = safe_num(shares)

    if (
        not np.isfinite(fcf)
        or not np.isfinite(shares)
        or fcf <= 0
        or shares <= 0
    ):

        return np.nan

    discount_rate = max(
        discount_rate,
        terminal_growth + 0.005
    )

    pv = 0

    current_fcf = fcf

    for year in range(1, 6):

        current_fcf *= (
            1 + growth
        )

        pv += (
            current_fcf
            / (1 + discount_rate) ** year
        )

    terminal_value = (

        current_fcf
        * (1 + terminal_growth)
        / (discount_rate - terminal_growth)

    )

    pv_terminal = (
        terminal_value
        / (1 + discount_rate) ** 5
    )

    enterprise_value = (
        pv + pv_terminal
    )

    return (
        enterprise_value
        / shares
    )


def dcf_analysis(info, fundamental):

    fcf = fundamental[
        "free_cash_flow"
    ]

    shares = info_value(
        info,
        "sharesOutstanding"
    )

    beta = fundamental[
        "beta"
    ]

    growth = fundamental[
        "earnings_growth"
    ]

    if not np.isfinite(growth):

        growth = fundamental[
            "revenue_growth"
        ]

    if not np.isfinite(growth):

        growth = 0.08

    growth = float(
        np.clip(
            growth,
            -0.02,
            0.18
        )
    )

    if np.isfinite(beta):

        discount = (
            0.085
            + np.clip(
                beta - 1,
                -0.4,
                1.2
            ) * 0.025
        )

    else:

        discount = 0.10

    discount = float(
        np.clip(
            discount,
            0.085,
            0.14
        )
    )

    bear = dcf_per_share(
        fcf,
        shares,
        np.clip(
            growth - 0.05,
            -0.05,
            0.12
        ),
        discount + 0.015,
        0.025
    )

    base = dcf_per_share(
        fcf,
        shares,
        np.clip(
            growth,
            0.02,
            0.15
        ),
        discount,
        0.03
    )

    bull = dcf_per_share(
        fcf,
        shares,
        np.clip(
            growth + 0.04,
            0.04,
            0.20
        ),
        max(
            discount - 0.01,
            0.08
        ),
        0.035
    )

    return {

        "bear": bear,

        "base": base,

        "bull": bull,

        "growth": growth,

        "discount": discount,

    }


# ============================================================
# MULTI-MODEL VALUATION
# ============================================================

def valuation_analysis(
    price,
    fundamental,
    dcf
):

    anchors = []

    # DCF

    if np.isfinite(
        dcf["base"]
    ):

        anchors.append(
            (
                "DCF",
                dcf["base"],
                0.45
            )
        )

    # Forward EPS

    forward_eps = fundamental[
        "forward_eps"
    ]

    earnings_growth = fundamental[
        "earnings_growth"
    ]

    revenue_growth = fundamental[
        "revenue_growth"
    ]

    if np.isfinite(
        forward_eps
    ):

        growth = earnings_growth

        if not np.isfinite(growth):

            growth = revenue_growth

        if not np.isfinite(growth):

            growth = 0.10

        target_pe = np.clip(
            18 + growth * 35,
            15,
            32
        )

        eps_value = (
            forward_eps
            * target_pe
        )

        anchors.append(
            (
                "Forward EPS × P/E",
                eps_value,
                0.35
            )
        )

    elif np.isfinite(
        fundamental["eps"]
    ):

        eps_value = (
            fundamental["eps"]
            * 20
        )

        anchors.append(
            (
                "EPS × 20",
                eps_value,
                0.20
            )
        )

    if not anchors:

        return {

            "fair": np.nan,

            "cheap": np.nan,

            "low": np.nan,

            "high": np.nan,

            "expensive": np.nan,

            "upside": np.nan,

            "anchors": [],

        }

    total_weight = sum(
        item[2]
        for item in anchors
    )

    fair_value = sum(
        item[1] * item[2]
        for item in anchors
    ) / total_weight

    return {

        "fair": fair_value,

        "cheap": fair_value * 0.78,

        "low": fair_value * 0.90,

        "high": fair_value * 1.10,

        "expensive": fair_value * 1.25,

        "upside": safe_div(
            fair_value,
            price
        ) - 1,

        "anchors": anchors,

    }


# ============================================================
# RISK ENGINE
# ============================================================

def risk_analysis(
    fundamental,
    technical
):

    risk = 30

    reasons = []

    beta = fundamental[
        "beta"
    ]

    volatility = technical[
        "volatility"
    ]

    debt_equity = fundamental[
        "debt_equity"
    ]

    current_ratio = fundamental[
        "current_ratio"
    ]

    if np.isfinite(beta):

        if beta > 1.5:

            risk += 18

            reasons.append(
                "Beta较高"
            )

        elif beta < 0.8:

            risk -= 6

    if np.isfinite(
        volatility
    ):

        if volatility > 0.60:

            risk += 18

            reasons.append(
                "近期波动率较高"
            )

        elif volatility < 0.25:

            risk -= 5

    if np.isfinite(
        debt_equity
    ):

        if debt_equity > 200:

            risk += 15

            reasons.append(
                "负债权益比较高"
            )

        elif debt_equity < 50:

            risk -= 7

    if np.isfinite(
        current_ratio
    ):

        if current_ratio < 1:

            risk += 12

            reasons.append(
                "流动比率低于1"
            )

    if not np.isfinite(
        fundamental[
            "free_cash_flow"
        ]
    ):

        risk += 5

        reasons.append(
            "自由现金流数据不足"
        )

    risk = float(
        np.clip(
            risk,
            0,
            100
        )
    )

    return risk, reasons


# ============================================================
# SIMON SCORE 3.0
# ============================================================

def calculate_simon_score(
    technical,
    fundamental,
    valuation,
    risk
):

    technical_score = (
        technical["score"]
    )

    fundamental_score = (
        fundamental["score"]
    )

    if np.isfinite(
        valuation["upside"]
    ):

        valuation_component = (
            50
            + np.clip(
                valuation["upside"] * 100,
                -35,
                35
            )
        )

    else:

        valuation_component = (
            fundamental[
                "valuation_score"
            ]
        )

    risk_component = (
        100 - risk
    )

    score = (

        technical_score * 0.25

        + fundamental_score * 0.40

        + valuation_component * 0.20

        + risk_component * 0.15

    )

    score = float(
        np.clip(
            score,
            0,
            100
        )
    )

    return score


# ============================================================
# PRICE ZONE
# ============================================================

def price_zone(
    price,
    valuation
):

    fair = valuation["fair"]

    if (
        not np.isfinite(price)
        or not np.isfinite(fair)
    ):

        return (
            "⚪ 数据不足",
            "暂无足够估值数据"
        )

    if price <= valuation["cheap"]:

        return (
            "🟢 白菜价",
            "深度安全边际"
        )

    if price <= valuation["low"]:

        return (
            "🟢 买入区",
            "偏便宜，可考虑分批"
        )

    if price <= valuation["high"]:

        return (
            "🔵 合理区",
            "适合正常持有 / 观察"
        )

    if price <= valuation["expensive"]:

        return (
            "🟡 偏贵区",
            "等待更好的价格"
        )

    return (
        "🔴 高估区",
        "安全边际不足"
    )


# ============================================================
# FULL ANALYSIS
# ============================================================

def analyze_stock(
    ticker,
    period="1y"
):

    history = get_history(
        ticker,
        period
    )

    if history.empty:

        return None

    info, financials, balance, cashflow = (
        get_company_data(ticker)
    )

    technical, technical_df = (
        technical_analysis(history)
    )

    fundamental = fundamental_analysis(
        info,
        financials,
        balance,
        cashflow
    )

    dcf = dcf_analysis(
        info,
        fundamental
    )

    valuation = valuation_analysis(
        technical["price"],
        fundamental,
        dcf
    )

    risk, risk_reasons = risk_analysis(
        fundamental,
        technical
    )

    simon_score = calculate_simon_score(
        technical,
        fundamental,
        valuation,
        risk
    )

    zone, zone_description = price_zone(
        technical["price"],
        valuation
    )

    return {

        "ticker": ticker,

        "info": info,

        "history": history,

        "technical_df": technical_df,

        "technical": technical,

        "fundamental": fundamental,

        "dcf": dcf,

        "valuation": valuation,

        "risk": risk,

        "risk_reasons": risk_reasons,

        "simon_score": simon_score,

        "zone": zone,

        "zone_description": zone_description,

    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🔎 Simon Stock V9")

    ticker = st.text_input(
        "股票代码",
        "AAPL"
    ).strip().upper()

    period = st.selectbox(
        "行情周期",
        [
            "6mo",
            "1y",
            "2y",
            "5y",
            "max",
        ],
        index=1
    )

    if st.button(
        "🚀 深度分析",
        use_container_width=True
    ):

        st.cache_data.clear()

        st.rerun()

    st.divider()

    st.subheader("⭐ 自选股")

    new_stock = st.text_input(
        "添加股票",
        placeholder="AMD"
    )

    if st.button("添加"):

        symbol = (
            new_stock
            .strip()
            .upper()
        )

        if (
            symbol
            and symbol not in st.session_state.watchlist
        ):

            st.session_state.watchlist.append(
                symbol
            )

            st.rerun()

    if st.session_state.watchlist:

        remove_stock = st.selectbox(
            "删除股票",
            ["不删除"]
            + st.session_state.watchlist
        )

        if st.button("删除"):

            if remove_stock != "不删除":

                st.session_state.watchlist.remove(
                    remove_stock
                )

                st.rerun()

    st.divider()

    st.subheader("⚔️ 股票 PK")

    pk_default = [
        "AAPL",
        "GOOGL",
        "NVDA",
    ]

    pk1 = st.text_input(
        "股票 A",
        pk_default[0]
    ).upper()

    pk2 = st.text_input(
        "股票 B",
        pk_default[1]
    ).upper()

    pk3 = st.text_input(
        "股票 C",
        pk_default[2]
    ).upper()


# ============================================================
# MAIN ANALYSIS
# ============================================================

if not ticker:

    st.warning(
        "请输入股票代码。"
    )

    st.stop()


with st.spinner(
    f"正在分析 {ticker} ..."
):

    result = analyze_stock(
        ticker,
        period
    )


if result is None:

    st.error(
        f"无法获取 {ticker} 的数据，请检查股票代码。"
    )

    st.stop()


info = result["info"]

technical = result["technical"]

fundamental = result["fundamental"]

valuation = result["valuation"]

dcf = result["dcf"]

risk = result["risk"]

score = result["simon_score"]

zone = result["zone"]

zone_description = result[
    "zone_description"
]

company_name = (
    info.get("longName")
    or info.get("shortName")
    or ticker
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📈 SIMON STOCK V9</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'US Stock Research Terminal · '
    'Fundamental + Technical + Valuation + Risk'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    f"## {ticker} · {company_name}"
)

st.caption(
    f"{info.get('sector', 'Unknown')} · "
    f"{info.get('industry', 'Unknown')}"
)


# ============================================================
# TOP METRICS
# ============================================================

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "当前价格",
    money(
        technical["price"]
    )
)

c2.metric(
    "Simon Score",
    f"{score:.0f}/100"
)

c3.metric(
    "评级",
    score_label(score)
)

c4.metric(
    "行动",
    score_action(score)
)

c5.metric(
    "风险",
    f"{risk:.0f}/100"
)


# ============================================================
# PRICE ZONE
# ============================================================

st.subheader(
    "🎯 Simon Price Zone"
)

p1, p2, p3, p4, p5 = st.columns(5)

p1.metric(
    "白菜价",
    money(
        valuation["cheap"]
    )
)

p2.metric(
    "买入区",
    money(
        valuation["low"]
    )
)

p3.metric(
    "合理价值",
    money(
        valuation["fair"]
    )
)

p4.metric(
    "合理高位",
    money(
        valuation["high"]
    )
)

p5.metric(
    "高估参考",
    money(
        valuation["expensive"]
    )
)

st.info(
    f"{zone} · {zone_description}"
)


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "📊 总览",
        "📈 技术面",
        "🏢 基本面",
        "💰 估值",
        "🚨 风险",
        "⭐ 自选股",
        "⚔️ 股票PK",
        "💼 Portfolio",
        "🧠 Research Report",
        "📚 原始数据",
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

with tabs[0]:

    left, right = st.columns(
        [1.4, 1]
    )

    with left:

        st.subheader(
            "📈 Price Trend"
        )

        chart_df = result[
            "technical_df"
        ][
            [
                "Close",
                "MA20",
                "MA50",
                "MA200",
            ]
        ].dropna(
            how="all"
        )

        st.line_chart(
            chart_df
        )

        st.subheader(
            "📊 Volume"
        )

        st.bar_chart(
            result[
                "technical_df"
            ]["Volume"]
        )

    with right:

        st.subheader(
            "🧠 Simon Score"
        )

        score_df = pd.DataFrame(
            {
                "Score": [

                    technical["score"],

                    fundamental[
                        "growth_score"
                    ],

                    fundamental[
                        "quality_score"
                    ],

                    fundamental[
                        "balance_score"
                    ],

                    fundamental[
                        "cashflow_score"
                    ],

                    fundamental[
                        "valuation_score"
                    ],

                    100 - risk,

                ]
            },

            index=[
                "技术面",
                "成长性",
                "盈利质量",
                "财务健康",
                "现金流",
                "估值",
                "风险调整",
            ]
        )

        st.bar_chart(
            score_df
        )

        st.metric(
            "合理价值潜在空间",
            percent(
                valuation["upside"]
            )
        )

        st.metric(
            "Beta",
            number(
                fundamental["beta"]
            )
        )


# ============================================================
# TECHNICAL
# ============================================================

with tabs[1]:

    st.subheader(
        "📈 Technical Analysis"
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "RSI",
        number(
            technical["rsi"]
        )
    )

    b.metric(
        "MA20",
        money(
            technical["ma20"]
        )
    )

    c.metric(
        "MA50",
        money(
            technical["ma50"]
        )
    )

    d.metric(
        "MA200",
        money(
            technical["ma200"]
        )
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "MACD",
        number(
            technical["macd"]
        )
    )

    b.metric(
        "MACD Signal",
        number(
            technical["macd_signal"]
        )
    )

    c.metric(
        "Volatility",
        percent(
            technical["volatility"]
        )
    )

    d.metric(
        "Technical Score",
        f"{technical['score']:.0f}/100"
    )

    st.progress(
        int(
            technical["score"]
        )
    )

    st.subheader(
        "技术面信号"
    )

    for reason in technical[
        "reasons"
    ]:

        st.write(
            "• " + reason
        )


# ============================================================
# FUNDAMENTAL
# ============================================================

with tabs[2]:

    st.subheader(
        "🏢 Fundamental Analysis"
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Revenue",
        compact_number(
            fundamental["revenue"]
        )
    )

    b.metric(
        "EPS",
        number(
            fundamental["eps"]
        )
    )

    c.metric(
        "Forward EPS",
        number(
            fundamental["forward_eps"]
        )
    )

    d.metric(
        "Free Cash Flow",
        compact_number(
            fundamental[
                "free_cash_flow"
            ]
        )
    )

    fundamental_df = pd.DataFrame(
        {
            "指标": [

                "Revenue Growth",

                "Earnings Growth",

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

                "EV / EBITDA",

                "Dividend Yield",

            ],

            "数值": [

                percent(
                    fundamental[
                        "revenue_growth"
                    ]
                ),

                percent(
                    fundamental[
                        "earnings_growth"
                    ]
                ),

                percent(
                    fundamental[
                        "gross_margin"
                    ]
                ),

                percent(
                    fundamental[
                        "operating_margin"
                    ]
                ),

                percent(
                    fundamental[
                        "profit_margin"
                    ]
                ),

                percent(
                    fundamental[
                        "roe"
                    ]
                ),

                percent(
                    fundamental[
                        "roa"
                    ]
                ),

                number(
                    fundamental[
                        "debt_equity"
                    ]
                ),

                number(
                    fundamental[
                        "current_ratio"
                    ]
                ),

                number(
                    fundamental["pe"]
                ),

                number(
                    fundamental[
                        "forward_pe"
                    ]
                ),

                number(
                    fundamental["peg"]
                ),

                number(
                    fundamental["ps"]
                ),

                number(
                    fundamental["pb"]
                ),

                number(
                    fundamental[
                        "ev_ebitda"
                    ]
                ),

                percent(
                    fundamental[
                        "dividend_yield"
                    ]
                ),

            ]
        }
    )

    st.dataframe(
        fundamental_df,
        hide_index=True,
        use_container_width=True
    )

    st.subheader(
        "Fundamental Score"
    )

    fundamental_score_df = pd.DataFrame(
        {
            "Score": [

                fundamental[
                    "growth_score"
                ],

                fundamental[
                    "quality_score"
                ],

                fundamental[
                    "balance_score"
                ],

                fundamental[
                    "cashflow_score"
                ],

                fundamental[
                    "valuation_score"
                ],

            ]
        },

        index=[
            "成长性",
            "盈利质量",
            "财务健康",
            "现金流",
            "估值",
        ]
    )

    st.bar_chart(
        fundamental_score_df
    )


# ============================================================
# VALUATION
# ============================================================

with tabs[3]:

    st.subheader(
        "💰 Simon Valuation Engine"
    )

    a, b, c = st.columns(3)

    a.metric(
        "DCF Bear",
        money(
            dcf["bear"]
        )
    )

    b.metric(
        "DCF Base",
        money(
            dcf["base"]
        )
    )

    c.metric(
        "DCF Bull",
        money(
            dcf["bull"]
        )
    )

    st.divider()

    st.subheader(
        "🐻 / 🔵 / 🐂 Scenario"
    )

    scenario_df = pd.DataFrame(
        {
            "情景": [
                "🐻 Bear",
                "🔵 Base",
                "🐂 Bull",
            ],

            "估值": [
                money(
                    dcf["bear"]
                ),

                money(
                    dcf["base"]
                ),

                money(
                    dcf["bull"]
                ),
            ]
        }
    )

    st.dataframe(
        scenario_df,
        hide_index=True,
        use_container_width=True
    )

    st.subheader(
        "Simon Price Zone"
    )

    price_df = pd.DataFrame(
        {
            "区域": [

                "🟢 白菜价",

                "🟢 买入区",

                "🔵 合理价值",

                "🟡 合理高位",

                "🔴 高估参考",

            ],

            "价格": [

                money(
                    valuation["cheap"]
                ),

                money(
                    valuation["low"]
                ),

                money(
                    valuation["fair"]
                ),

                money(
                    valuation["high"]
                ),

                money(
                    valuation["expensive"]
                ),

            ]
        }
    )

    st.dataframe(
        price_df,
        hide_index=True,
        use_container_width=True
    )

    if np.isfinite(
        valuation["upside"]
    ):

        st.metric(
            "当前价格 → 合理价值",
            percent(
                valuation["upside"]
            )
        )

    st.caption(
        f"DCF基准增长率约 "
        f"{percent(dcf['growth'])}；"
        f"折现率约 "
        f"{percent(dcf['discount'])}。"
    )

    st.warning(
        "估值模型属于研究辅助模型，不是精确目标价。"
    )


# ============================================================
# RISK
# ============================================================

with tabs[4]:

    st.subheader(
        "🚨 Risk Radar"
    )

    risk_cols = st.columns(4)

    risk_cols[0].metric(
        "Risk Score",
        f"{risk:.0f}/100"
    )

    risk_cols[1].metric(
        "Beta",
        number(
            fundamental["beta"]
        )
    )

    risk_cols[2].metric(
        "Volatility",
        percent(
            technical["volatility"]
        )
    )

    risk_cols[3].metric(
        "Debt / Equity",
        number(
            fundamental[
                "debt_equity"
            ]
        )
    )

    st.progress(
        int(risk)
    )

    if risk >= 70:

        st.error(
            "⚠️ 风险较高"
        )

    elif risk >= 50:

        st.warning(
            "⚠️ 风险中等"
        )

    else:

        st.success(
            "✅ 风险相对可控"
        )

    if result[
        "risk_reasons"
    ]:

        st.subheader(
            "风险来源"
        )

        for reason in result[
            "risk_reasons"
        ]:

            st.write(
                "• " + reason
            )

    else:

        st.success(
            "当前模型没有检测到明显的高风险因素。"
        )


# ============================================================
# WATCHLIST
# ============================================================

with tabs[5]:

    st.subheader(
        "⭐ Simon Watchlist Scanner"
    )

    st.caption(
        "自动扫描自选股并按 Simon Score 排名。"
    )

    if st.button(
        "🔄 刷新扫描"
    ):

        st.cache_data.clear()

        st.rerun()

    watch_rows = []

    progress = st.progress(0)

    total = len(
        st.session_state.watchlist
    )

    for i, symbol in enumerate(
        st.session_state.watchlist
    ):

        try:

            data = analyze_stock(
                symbol,
                "6mo"
            )

            if data is not None:

                t = data["technical"]

                f = data["fundamental"]

                v = data["valuation"]

                watch_rows.append(
                    {

                        "Ticker":
                            symbol,

                        "Price":
                            money(
                                t["price"]
                            ),

                        "Simon Score":
                            round(
                                data[
                                    "simon_score"
                                ]
                            ),

                        "Technical":
                            round(
                                t["score"]
                            ),

                        "Fundamental":
                            round(
                                f["score"]
                            ),

                        "Risk":
                            round(
                                data["risk"]
                            ),

                        "Upside":
                            percent(
                                v["upside"]
                            ),

                        "Zone":
                            data["zone"],

                        "Action":
                            score_action(
                                data[
                                    "simon_score"
                                ]
                            ),

                    }
                )

        except Exception:

            pass

        progress.progress(
            int(
                (i + 1)
                / max(total, 1)
                * 100
            )
        )

    progress.empty()

    if watch_rows:

        watch_df = pd.DataFrame(
            watch_rows
        )

        watch_df = watch_df.sort_values(
            "Simon Score",
            ascending=False
        )

        st.dataframe(
            watch_df,
            hide_index=True,
            use_container_width=True
        )

    else:

        st.info(
            "暂时没有成功加载自选股。"
        )


# ============================================================
# STOCK BATTLE
# ============================================================

with tabs[6]:

    st.subheader(
        "⚔️ Simon Stock Battle"
    )

    st.write(
        f"比较："
        f"**{pk1} vs {pk2} vs {pk3}**"
    )

    battle_rows = []

    for symbol in [
        pk1,
        pk2,
        pk3
    ]:

        if not symbol:
            continue

        try:

            data = analyze_stock(
                symbol,
                "1y"
            )

            if data is None:
                continue

            t = data["technical"]

            f = data["fundamental"]

            v = data["valuation"]

            battle_rows.append(
                {

                    "Ticker":
                        symbol,

                    "Price":
                        money(
                            t["price"]
                        ),

                    "Simon Score":
                        round(
                            data[
                                "simon_score"
                            ]
                        ),

                    "Growth":
                        round(
                            f[
                                "growth_score"
                            ]
                        ),

                    "Quality":
                        round(
                            f[
                                "quality_score"
                            ]
                        ),

                    "Valuation":
                        round(
                            f[
                                "valuation_score"
                            ]
                        ),

                    "Technical":
                        round(
                            t["score"]
                        ),

                    "Risk":
                        round(
                            data["risk"]
                        ),

                    "Fair Value":
                        money(
                            v["fair"]
                        ),

                    "Upside":
                        percent(
                            v["upside"]
                        ),

                }
            )

        except Exception:

            pass

    if battle_rows:

        battle_df = pd.DataFrame(
            battle_rows
        )

        battle_df = battle_df.sort_values(
            "Simon Score",
            ascending=False
        )

        st.dataframe(
            battle_df,
            hide_index=True,
            use_container_width=True
        )

        winner = battle_df.iloc[0]

        st.success(
            f"🏆 当前模型第一名："
            f"{winner['Ticker']} · "
            f"Simon Score "
            f"{winner['Simon Score']}"
        )

    else:

        st.warning(
            "无法完成股票 PK。"
        )


# ============================================================
# PORTFOLIO
# ============================================================

with tabs[7]:

    st.subheader(
        "💼 Portfolio Analyzer"
    )

    st.caption(
        "输入你的实际持仓成本与数量，分析组合集中度。"
    )

    with st.form(
        "portfolio_form"
    ):

        portfolio_ticker = st.text_input(
            "股票代码",
            "AAPL"
        ).upper()

        portfolio_shares = st.number_input(
            "持股数量",
            min_value=0.0,
            value=1.0,
            step=1.0
        )

        portfolio_cost = st.number_input(
            "平均成本",
            min_value=0.0,
            value=300.0,
            step=1.0
        )

        submitted = st.form_submit_button(
            "加入组合"
        )

        if submitted:

            st.session_state.portfolio.append(
                {

                    "Ticker":
                        portfolio_ticker,

                    "Shares":
                        portfolio_shares,

                    "Cost":
                        portfolio_cost,

                }
            )

            st.success(
                "已加入 Portfolio"
            )

    if st.session_state.portfolio:

        portfolio_rows = []

        total_value = 0

        total_cost = 0

        for item in (
            st.session_state.portfolio
        ):

            try:

                data = analyze_stock(
                    item["Ticker"],
                    "6mo"
                )

                if data is None:
                    continue

                price = data[
                    "technical"
                ]["price"]

                value = (
                    price
                    * item["Shares"]
                )

                cost = (
                    item["Cost"]
                    * item["Shares"]
                )

                profit = (
                    value - cost
                )

                total_value += value

                total_cost += cost

                portfolio_rows.append(
                    {

                        "Ticker":
                            item["Ticker"],

                        "Shares":
                            item["Shares"],

                        "Avg Cost":
                            money(
                                item["Cost"]
                            ),

                        "Current":
                            money(
                                price
                            ),

                        "Market Value":
                            money(
                                value
                            ),

                        "P/L":
                            money(
                                profit
                            ),

                        "Return":
                            percent(
                                safe_div(
                                    profit,
                                    cost
                                )
                            ),

                    }
                )

            except Exception:

                pass

        if portfolio_rows:

            portfolio_df = pd.DataFrame(
                portfolio_rows
            )

            st.dataframe(
                portfolio_df,
                hide_index=True,
                use_container_width=True
            )

            st.divider()

            total_profit = (
                total_value
                - total_cost
            )

            a, b, c = st.columns(3)

            a.metric(
                "Portfolio Value",
                money(
                    total_value
                )
            )

            b.metric(
                "Total Cost",
                money(
                    total_cost
                )
            )

            c.metric(
                "Total P/L",
                money(
                    total_profit
                )
            )

        if st.button(
            "🗑️ 清空 Portfolio"
        ):

            st.session_state.portfolio = []

            st.rerun()

    else:

        st.info(
            "还没有持仓。"
        )


# ============================================================
# RESEARCH REPORT
# ============================================================

with tabs[8]:

    st.subheader(
        "🧠 Simon Research Report"
    )

    st.markdown(
        f"""
## {ticker} — {company_name}

### 🧠 Simon Score

**{score:.0f}/100 · {score_label(score)} · {score_action(score)}**

---

### 💰 估值

当前价格：

**{money(technical["price"])}**

Simon合理价值：

**{money(valuation["fair"])}**

当前估值空间：

**{percent(valuation["upside"])}**

当前价格区域：

**{zone}**

{zone_description}

---

### 📊 技术面

Technical Score：

**{technical["score"]:.0f}/100**

RSI：

**{number(technical["rsi"])}**

MA20：

**{money(technical["ma20"])}**

MA50：

**{money(technical["ma50"])}**

MA200：

**{money(technical["ma200"])}**

---

### 🏢 基本面

Fundamental Score：

**{fundamental["score"]:.0f}/100**

成长性：

**{fundamental["growth_score"]:.0f}/100**

盈利质量：

**{fundamental["quality_score"]:.0f}/100**

财务健康：

**{fundamental["balance_score"]:.0f}/100**

现金流：

**{fundamental["cashflow_score"]:.0f}/100**

---

### 🚨 风险

Risk Score：

**{risk:.0f}/100**

Beta：

**{number(fundamental["beta"])}**

---

### 🐻 / 🔵 / 🐂 三情景

Bear：

**{money(dcf["bear"])}**

Base：

**{money(dcf["base"])}**

Bull：

**{money(dcf["bull"])}**

---

### 🎯 Simon结论

**{score_action(score)}**

模型当前认为：

**{zone}**

---

> ⚠️ Simon Stock V9 是投资研究辅助工具，不构成投资建议。
> Yahoo Finance 数据可能存在延迟、缺失或字段变化。
"""
    )


# ============================================================
# RAW DATA
# ============================================================

with tabs[9]:

    st.subheader(
        "📚 Raw Yahoo Finance Data"
    )

    basic_keys = [

        "symbol",

        "shortName",

        "longName",

        "sector",

        "industry",

        "country",

        "marketCap",

        "enterpriseValue",

        "trailingPE",

        "forwardPE",

        "pegRatio",

        "priceToBook",

        "beta",

        "fiftyTwoWeekHigh",

        "fiftyTwoWeekLow",

    ]

    raw_info = {}

    for key in basic_keys:

        if key in info:

            raw_info[key] = info[key]

    st.json(
        raw_info
    )

    st.subheader(
        "Historical Data"
    )

    st.dataframe(
        result["history"].tail(150),
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Simon Stock V9 · "
    "US Stock Research Terminal · "
    "Data via Yahoo Finance / yfinance · "
    + datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
)

st.caption(
    "Research tool only — "
    "not financial advice."
)