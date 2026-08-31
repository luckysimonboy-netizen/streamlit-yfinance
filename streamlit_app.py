import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ============================================================
# SIMON STOCK V7.0
# AI-Native Investment Research Engine
#
# Free core engine
# No OpenAI API required
# Data source: Yahoo Finance via yfinance
#
# IMPORTANT:
# This is a research / education tool.
# It is NOT financial advice.
# ============================================================

st.set_page_config(
    page_title="Simon Stock V7.0",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1500px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 30px;
        border-radius: 24px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 25px;
        background: linear-gradient(
            135deg,
            rgba(70,90,180,0.14),
            rgba(140,70,180,0.08)
        );
    }

    .hero-title {
        font-size: 46px;
        font-weight: 900;
        line-height: 1.1;
    }

    .hero-subtitle {
        opacity: 0.68;
        margin-top: 8px;
    }

    .decision {
        font-size: 30px;
        font-weight: 900;
    }

    .score {
        font-size: 60px;
        font-weight: 900;
    }

    .card {
        padding: 20px;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,0.20);
        background: rgba(128,128,128,0.05);
        margin-bottom: 15px;
    }

    .small {
        opacity: 0.62;
        font-size: 13px;
    }

    .pill {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        margin-right: 5px;
        border: 1px solid rgba(128,128,128,0.25);
        font-size: 13px;
    }

    .positive {
        font-weight: 800;
    }

    .negative {
        font-weight: 800;
    }

    .neutral {
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# BASIC HELPERS
# ============================================================

def safe_float(value, default=np.nan):
    try:
        if value is None:
            return default

        result = float(value)

        if np.isnan(result) or np.isinf(result):
            return default

        return result

    except Exception:
        return default


def safe_div(a, b, default=np.nan):
    a = safe_float(a)
    b = safe_float(b)

    if np.isnan(a) or np.isnan(b) or b == 0:
        return default

    return a / b


def clamp(value, low=0, high=100):
    value = safe_float(value, 50)

    return max(
        low,
        min(high, value)
    )


def percentage(value):
    value = safe_float(value)

    if np.isnan(value):
        return "N/A"

    return f"{value * 100:.1f}%"


def dollar(value):
    value = safe_float(value)

    if np.isnan(value):
        return "N/A"

    abs_value = abs(value)

    if abs_value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"

    if abs_value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"

    if abs_value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"

    return f"${value:,.2f}"


def get_value(info, *keys):
    for key in keys:

        if key not in info:
            continue

        value = safe_float(
            info.get(key)
        )

        if not np.isnan(value):
            return value

    return np.nan


def fmt_number(value, digits=2):
    value = safe_float(value)

    if np.isnan(value):
        return "N/A"

    return f"{value:.{digits}f}"


# ============================================================
# DATA ENGINE
# ============================================================

@st.cache_data(
    ttl=300,
    show_spinner=False
)
def load_stock(symbol, period):

    ticker = yf.Ticker(symbol)

    history = ticker.history(
        period=period,
        interval="1d",
        auto_adjust=False
    )

    try:
        info = ticker.info
    except Exception:
        info = {}

    try:
        news = ticker.news
    except Exception:
        news = []

    return history, info, news


# ============================================================
# DATA QUALITY ENGINE
# ============================================================

def data_quality(history, info):

    score = 100
    issues = []
    checks = []

    # Price history
    if history is None or history.empty:

        return {
            "score": 0,
            "confidence": "LOW",
            "issues": ["没有历史价格数据。"],
            "checks": []
        }

    checks.append("历史价格数据存在")

    if len(history) < 30:

        score -= 20

        issues.append(
            "历史价格样本较少。"
        )

    else:

        checks.append(
            "历史价格样本充足"
        )

    # Missing values
    if "Close" in history.columns:

        missing_ratio = (
            history["Close"].isna().mean()
        )

        if missing_ratio > 0.05:

            score -= 15

            issues.append(
                "收盘价存在较明显缺失。"
            )

        else:

            checks.append(
                "价格缺失率较低"
            )

    # Fundamental data
    important_fields = [
        "marketCap",
        "revenueGrowth",
        "profitMargins",
        "returnOnEquity",
        "freeCashflow",
        "trailingPE"
    ]

    available = sum(
        1
        for field in important_fields
        if not np.isnan(
            get_value(info, field)
        )
    )

    fundamental_ratio = (
        available /
        len(important_fields)
    )

    if fundamental_ratio < 0.50:

        score -= 20

        issues.append(
            "关键基本面数据不完整。"
        )

    elif fundamental_ratio < 0.75:

        score -= 10

        issues.append(
            "部分基本面数据缺失。"
        )

    else:

        checks.append(
            "主要基本面指标可用"
        )

    score = int(
        clamp(score)
    )

    if score >= 85:
        confidence = "HIGH"

    elif score >= 70:
        confidence = "MEDIUM"

    else:
        confidence = "LOW"

    return {
        "score": score,
        "confidence": confidence,
        "issues": issues,
        "checks": checks
    }


# ============================================================
# TECHNICAL ENGINE
# ============================================================

def technical_analysis(history):

    result = {}

    if history is None or history.empty:
        return result

    close = (
        history["Close"]
        .dropna()
        .astype(float)
    )

    if close.empty:
        return result

    price = safe_float(
        close.iloc[-1]
    )

    result["price"] = price

    # Moving averages
    for window in [20, 50, 100, 200]:

        if len(close) >= window:

            result[
                f"sma{window}"
            ] = safe_float(
                close
                .rolling(window)
                .mean()
                .iloc[-1]
            )

    # EMA
    for window in [20, 50]:

        if len(close) >= window:

            result[
                f"ema{window}"
            ] = safe_float(
                close
                .ewm(
                    span=window,
                    adjust=False
                )
                .mean()
                .iloc[-1]
            )

    # RSI
    if len(close) >= 15:

        delta = close.diff()

        gain = delta.clip(
            lower=0
        )

        loss = -delta.clip(
            upper=0
        )

        avg_gain = (
            gain
            .rolling(14)
            .mean()
        )

        avg_loss = (
            loss
            .rolling(14)
            .mean()
        )

        rs = (
            avg_gain /
            avg_loss.replace(
                0,
                np.nan
            )
        )

        rsi = (
            100 -
            100 /
            (1 + rs)
        )

        result["rsi"] = safe_float(
            rsi.iloc[-1]
        )

    # Returns
    for days, name in [
        (5, "return5"),
        (20, "return20"),
        (60, "return60"),
        (252, "return1y")
    ]:

        if len(close) > days:

            old_price = safe_float(
                close.iloc[-days - 1]
            )

            result[name] = (
                price /
                old_price -
                1
            )

    # Volatility
    returns = close.pct_change().dropna()

    if len(returns) >= 20:

        result["volatility20"] = (
            safe_float(
                returns
                .tail(20)
                .std()
            ) *
            np.sqrt(252)
        )

    # Drawdown
    rolling_high = close.cummax()

    drawdown = (
        close /
        rolling_high -
        1
    )

    result["current_drawdown"] = safe_float(
        drawdown.iloc[-1]
    )

    result["max_drawdown"] = safe_float(
        drawdown.min()
    )

    # 52-week range
    if len(close) >= 252:

        last_year = close.tail(252)

        result["52w_high"] = safe_float(
            last_year.max()
        )

        result["52w_low"] = safe_float(
            last_year.min()
        )

        result["distance_from_52w_high"] = (
            price /
            result["52w_high"] -
            1
        )

    return result


# ============================================================
# FUNDAMENTAL ENGINE
# ============================================================

def fundamental_metrics(info):

    metrics = {

        "revenue_growth":
            get_value(
                info,
                "revenueGrowth"
            ),

        "earnings_growth":
            get_value(
                info,
                "earningsGrowth"
            ),

        "profit_margin":
            get_value(
                info,
                "profitMargins"
            ),

        "operating_margin":
            get_value(
                info,
                "operatingMargins"
            ),

        "gross_margin":
            get_value(
                info,
                "grossMargins"
            ),

        "roe":
            get_value(
                info,
                "returnOnEquity"
            ),

        "roa":
            get_value(
                info,
                "returnOnAssets"
            ),

        "debt_equity":
            get_value(
                info,
                "debtToEquity"
            ),

        "current_ratio":
            get_value(
                info,
                "currentRatio"
            ),

        "quick_ratio":
            get_value(
                info,
                "quickRatio"
            ),

        "fcf":
            get_value(
                info,
                "freeCashflow"
            ),

        "operating_cashflow":
            get_value(
                info,
                "operatingCashflow"
            ),

        "pe":
            get_value(
                info,
                "trailingPE"
            ),

        "forward_pe":
            get_value(
                info,
                "forwardPE"
            ),

        "peg":
            get_value(
                info,
                "pegRatio"
            ),

        "price_to_sales":
            get_value(
                info,
                "priceToSalesTrailing12Months"
            ),

        "price_to_book":
            get_value(
                info,
                "priceToBook"
            ),

        "beta":
            get_value(
                info,
                "beta"
            ),

        "market_cap":
            get_value(
                info,
                "marketCap"
            ),

        "enterprise_value":
            get_value(
                info,
                "enterpriseValue"
            ),

        "shares":
            get_value(
                info,
                "sharesOutstanding"
            ),

        "free_cashflow_yield":
            np.nan
    }

    market_cap = metrics[
        "market_cap"
    ]

    fcf = metrics[
        "fcf"
    ]

    if (
        not np.isnan(market_cap)
        and market_cap > 0
        and not np.isnan(fcf)
    ):

        metrics[
            "free_cashflow_yield"
        ] = (
            fcf /
            market_cap
        )

    return metrics


# ============================================================
# SCORE ENGINE
# ============================================================

def score_growth(m):

    score = 50

    revenue = m[
        "revenue_growth"
    ]

    earnings = m[
        "earnings_growth"
    ]

    if not np.isnan(revenue):

        if revenue >= 0.30:
            score += 30

        elif revenue >= 0.20:
            score += 22

        elif revenue >= 0.10:
            score += 12

        elif revenue >= 0:
            score += 2

        else:
            score -= 25

    if not np.isnan(earnings):

        if earnings >= 0.30:
            score += 25

        elif earnings >= 0.20:
            score += 18

        elif earnings >= 0.10:
            score += 10

        elif earnings >= 0:
            score += 2

        else:
            score -= 25

    return clamp(score)


def score_profitability(m):

    score = 50

    roe = m["roe"]
    margin = m["profit_margin"]
    operating = m["operating_margin"]
    gross = m["gross_margin"]

    if not np.isnan(roe):

        if roe >= 0.30:
            score += 25

        elif roe >= 0.20:
            score += 18

        elif roe >= 0.10:
            score += 8

        elif roe < 0:
            score -= 25

    if not np.isnan(margin):

        if margin >= 0.25:
            score += 20

        elif margin >= 0.15:
            score += 12

        elif margin >= 0.05:
            score += 4

        elif margin < 0:
            score -= 20

    if not np.isnan(operating):

        if operating >= 0.25:
            score += 15

        elif operating >= 0.15:
            score += 8

        elif operating < 0:
            score -= 15

    if not np.isnan(gross):

        if gross >= 0.60:
            score += 10

        elif gross >= 0.40:
            score += 5

    return clamp(score)


def score_financial(m):

    score = 50

    debt = m[
        "debt_equity"
    ]

    current = m[
        "current_ratio"
    ]

    fcf = m[
        "fcf"
    ]

    cfo = m[
        "operating_cashflow"
    ]

    if not np.isnan(debt):

        if debt < 40:
            score += 20

        elif debt < 80:
            score += 12

        elif debt < 150:
            score += 0

        else:
            score -= 25

    if not np.isnan(current):

        if current >= 2:
            score += 15

        elif current >= 1:
            score += 5

        elif current < 0.7:
            score -= 15

    if not np.isnan(fcf):

        if fcf > 0:
            score += 15

        else:
            score -= 20

    if not np.isnan(cfo):

        if cfo > 0:
            score += 5

        else:
            score -= 10

    return clamp(score)


def score_capital_efficiency(m):

    score = 50

    roe = m["roe"]
    roa = m["roa"]

    if not np.isnan(roe):

        if roe >= 0.30:
            score += 25

        elif roe >= 0.20:
            score += 18

        elif roe >= 0.10:
            score += 8

        elif roe < 0:
            score -= 20

    if not np.isnan(roa):

        if roa >= 0.15:
            score += 20

        elif roa >= 0.08:
            score += 10

        elif roa < 0:
            score -= 15

    return clamp(score)


def score_valuation(m):

    score = 50

    pe = m["pe"]
    forward_pe = m["forward_pe"]
    peg = m["peg"]
    fcf_yield = m[
        "free_cashflow_yield"
    ]

    chosen_pe = forward_pe

    if np.isnan(chosen_pe):
        chosen_pe = pe

    if not np.isnan(chosen_pe):

        if chosen_pe <= 12:
            score += 30

        elif chosen_pe <= 18:
            score += 22

        elif chosen_pe <= 25:
            score += 12

        elif chosen_pe <= 35:
            score += 2

        elif chosen_pe <= 50:
            score -= 15

        else:
            score -= 30

    if not np.isnan(peg):

        if peg < 1:
            score += 15

        elif peg < 1.5:
            score += 8

        elif peg > 3:
            score -= 18

    if not np.isnan(fcf_yield):

        if fcf_yield >= 0.08:
            score += 20

        elif fcf_yield >= 0.05:
            score += 12

        elif fcf_yield >= 0.03:
            score += 5

        elif fcf_yield < 0.01:
            score -= 12

    return clamp(score)


def score_risk(m, t):

    score = 70

    beta = m["beta"]
    debt = m["debt_equity"]

    if not np.isnan(beta):

        if beta <= 1:
            score += 10

        elif beta <= 1.5:
            score += 0

        elif beta <= 2:
            score -= 10

        else:
            score -= 25

    if not np.isnan(debt):

        if debt > 200:
            score -= 20

        elif debt > 150:
            score -= 10

    max_dd = t.get(
        "max_drawdown",
        np.nan
    )

    if not np.isnan(max_dd):

        if max_dd > -0.20:
            score += 10

        elif max_dd > -0.40:
            score += 0

        elif max_dd > -0.60:
            score -= 10

        else:
            score -= 20

    volatility = t.get(
        "volatility20",
        np.nan
    )

    if not np.isnan(volatility):

        if volatility > 0.60:
            score -= 15

        elif volatility > 0.40:
            score -= 8

    return clamp(score)


# ============================================================
# BUSINESS QUALITY
# ============================================================

def score_business_quality(info, m):

    score = 50

    market_cap = m[
        "market_cap"
    ]

    revenue = m[
        "revenue_growth"
    ]

    margin = m[
        "profit_margin"
    ]

    roe = m[
        "roe"
    ]

    fcf = m[
        "fcf"
    ]

    # Scale is NOT treated as quality by itself.
    # It only gives a small stability bonus.

    if not np.isnan(market_cap):

        if market_cap >= 500e9:
            score += 8

        elif market_cap >= 100e9:
            score += 5

        elif market_cap >= 10e9:
            score += 2

    if not np.isnan(revenue):

        if revenue > 0.10:
            score += 8

        elif revenue < 0:
            score -= 8

    if not np.isnan(margin):

        if margin > 0.20:
            score += 10

        elif margin < 0:
            score -= 10

    if not np.isnan(roe):

        if roe > 0.20:
            score += 10

        elif roe < 0:
            score -= 10

    if not np.isnan(fcf):

        if fcf > 0:
            score += 8

        else:
            score -= 10

    return clamp(score)


# ============================================================
# MASTER SCORE
# ============================================================

def calculate_simon_score(
    info,
    metrics,
    technical
):

    scores = {}

    scores[
        "Business Quality"
    ] = score_business_quality(
        info,
        metrics
    )

    scores[
        "Growth"
    ] = score_growth(
        metrics
    )

    scores[
        "Profitability"
    ] = score_profitability(
        metrics
    )

    scores[
        "Financial Strength"
    ] = score_financial(
        metrics
    )

    scores[
        "Capital Efficiency"
    ] = score_capital_efficiency(
        metrics
    )

    scores[
        "Valuation"
    ] = score_valuation(
        metrics
    )

    scores[
        "Risk"
    ] = score_risk(
        metrics,
        technical
    )

    weights = {

        "Business Quality": 0.18,

        "Growth": 0.14,

        "Profitability": 0.16,

        "Financial Strength": 0.14,

        "Capital Efficiency": 0.10,

        "Valuation": 0.18,

        "Risk": 0.10
    }

    total = 0

    for key, weight in weights.items():

        total += (
            scores[key] *
            weight
        )

    return (
        int(round(total)),
        scores
    )


# ============================================================
# DCF ENGINE
# ============================================================

def dcf_scenario(
    fcf_per_share,
    growth,
    discount,
    terminal_growth
):

    if (
        np.isnan(fcf_per_share)
        or fcf_per_share <= 0
    ):
        return np.nan

    projected = fcf_per_share

    pv = 0

    for year in range(1, 6):

        projected *= (
            1 + growth
        )

        pv += (
            projected /
            ((1 + discount) ** year)
        )

    terminal = (
        projected *
        (1 + terminal_growth)
        /
        (
            discount -
            terminal_growth
        )
    )

    terminal_pv = (
        terminal /
        ((1 + discount) ** 5)
    )

    return max(
        0,
        pv + terminal_pv
    )


def dcf_model(info):

    fcf = get_value(
        info,
        "freeCashflow"
    )

    shares = get_value(
        info,
        "sharesOutstanding"
    )

    if (
        np.isnan(fcf)
        or np.isnan(shares)
        or fcf <= 0
        or shares <= 0
    ):
        return None

    fcf_per_share = (
        fcf /
        shares
    )

    earnings_growth = get_value(
        info,
        "earningsGrowth"
    )

    revenue_growth = get_value(
        info,
        "revenueGrowth"
    )

    growth_candidates = [
        x
        for x in [
            earnings_growth,
            revenue_growth
        ]
        if not np.isnan(x)
    ]

    if growth_candidates:

        base_growth = float(
            np.median(
                growth_candidates
            )
        )

    else:

        base_growth = 0.08

    base_growth = clamp(
        base_growth,
        -0.05,
        0.20
    ) / 100 if False else base_growth

    # Explicit bounds
    base_growth = max(
        -0.05,
        min(
            0.20,
            base_growth
        )
    )

    scenarios = {

        "Bear": {
            "growth":
                max(
                    -0.03,
                    base_growth - 0.06
                ),
            "discount": 0.10,
            "terminal":
                0.02
        },

        "Base": {
            "growth":
                max(
                    0.00,
                    base_growth - 0.02
                ),
            "discount": 0.085,
            "terminal":
                0.025
        },

        "Bull": {
            "growth":
                min(
                    0.20,
                    base_growth + 0.04
                ),
            "discount": 0.08,
            "terminal":
                0.03
        }
    }

    values = {}

    for name, assumption in scenarios.items():

        values[name] = dcf_scenario(
            fcf_per_share,
            assumption["growth"],
            assumption["discount"],
            assumption["terminal"]
        )

    return values


# ============================================================
# MULTIPLE VALUATION
# ============================================================

def multiple_fair_value(
    info,
    price
):

    pe = get_value(
        info,
        "trailingPE"
    )

    forward_pe = get_value(
        info,
        "forwardPE"
    )

    growth = get_value(
        info,
        "earningsGrowth"
    )

    values = []

    if not np.isnan(pe) and pe > 0:

        if not np.isnan(growth):

            if growth >= 0.25:
                target_pe = 30

            elif growth >= 0.15:
                target_pe = 27

            elif growth >= 0.08:
                target_pe = 24

            elif growth >= 0:
                target_pe = 20

            else:
                target_pe = 16

        else:

            target_pe = 22

        values.append(
            price *
            target_pe /
            pe
        )

    if (
        not np.isnan(forward_pe)
        and forward_pe > 0
    ):

        values.append(
            price *
            23 /
            forward_pe
        )

    if not values:
        return np.nan

    return float(
        np.median(values)
    )


# ============================================================
# FCF YIELD VALUATION
# ============================================================

def fcf_yield_value(
    info,
    price
):

    fcf = get_value(
        info,
        "freeCashflow"
    )

    market_cap = get_value(
        info,
        "marketCap"
    )

    if (
        np.isnan(fcf)
        or np.isnan(market_cap)
        or market_cap <= 0
        or fcf <= 0
    ):
        return np.nan

    current_yield = (
        fcf /
        market_cap
    )

    target_yield = 0.04

    if current_yield <= 0:
        return np.nan

    return (
        price *
        current_yield /
        target_yield
    )


# ============================================================
# FAIR VALUE ENGINE
# ============================================================

def fair_value_engine(
    info,
    price
):

    dcf = dcf_model(
        info
    )

    multiple = multiple_fair_value(
        info,
        price
    )

    fcf_value = fcf_yield_value(
        info,
        price
    )

    components = []

    if dcf is not None:

        base = safe_float(
            dcf.get(
                "Base",
                np.nan
            )
        )

        if not np.isnan(base):
            components.append(
                base
            )

    if not np.isnan(multiple):
        components.append(
            multiple
        )

    if not np.isnan(fcf_value):
        components.append(
            fcf_value
        )

    if not components:
        return None

    fair = float(
        np.median(
            components
        )
    )

    # Prevent absurd model outputs.
    fair = max(
        price * 0.40,
        min(
            price * 2.50,
            fair
        )
    )

    return {

        "fair": fair,

        "strong_buy":
            fair * 0.70,

        "buy":
            fair * 0.85,

        "expensive":
            fair * 1.15,

        "danger":
            fair * 1.35,

        "dcf":
            dcf,

        "multiple":
            multiple,

        "fcf_value":
            fcf_value,

        "components":
            components
    }


# ============================================================
# PRICE ATTRACTIVENESS
# ============================================================

def price_attractiveness(
    price,
    valuation
):

    if valuation is None:
        return np.nan

    fair = valuation[
        "fair"
    ]

    if np.isnan(fair) or fair <= 0:
        return np.nan

    upside = (
        fair /
        price -
        1
    )

    if upside >= 0.40:
        return 95

    if upside >= 0.25:
        return 85

    if upside >= 0.10:
        return 75

    if upside >= 0:
        return 65

    if upside >= -0.10:
        return 55

    if upside >= -0.25:
        return 40

    if upside >= -0.40:
        return 25

    return 10


# ============================================================
# VERDICT ENGINE
# ============================================================

def generate_verdict(
    score,
    price_score,
    price,
    valuation,
    risk_preference
):

    if valuation is None:

        if score >= 85:

            return (
                "🟡 GREAT BUSINESS / DATA LIMITED",
                "公司质量信号不错，但当前缺乏足够估值数据。"
            )

        if score >= 70:

            return (
                "🟡 WATCH",
                "基本面中等偏好，但估值证据不足。"
            )

        return (
            "🔴 WAIT",
            "当前证据不足以支持积极判断。"
        )

    fair = valuation[
        "fair"
    ]

    if np.isnan(price_score):

        return (
            "🟡 WATCH",
            "估值模型暂时无法形成可靠的价格判断。"
        )

    if (
        score >= 85
        and price_score >= 85
    ):

        return (
            "🟢 STRONG BUY ZONE",
            "公司质量高，同时当前价格提供了较好的安全边际。"
        )

    if (
        score >= 80
        and price_score >= 70
    ):

        return (
            "🟢 BUY ZONE",
            "公司质量与价格之间存在较好的平衡。"
        )

    if (
        score >= 80
        and price_score >= 50
    ):

        return (
            "🟡 GREAT COMPANY / WAIT",
            "公司质量很好，但价格没有提供足够安全边际。"
        )

    if (
        score >= 70
        and price_score >= 70
    ):

        return (
            "🟢 SELECTIVE BUY",
            "质量尚可且价格有一定吸引力，但需要控制仓位。"
        )

    if (
        score >= 70
        and price_score >= 45
    ):

        return (
            "🟡 WATCH",
            "公司存在一定价值，但当前并不形成强烈的赔率优势。"
        )

    if price > fair * 1.35:

        return (
            "🔴 AVOID / WAIT",
            "估值明显高于模型合理价值。"
        )

    return (
        "🟠 LOW CONVICTION",
        "当前风险收益比不足以形成高确定性判断。"
    )


# ============================================================
# MASTER COUNCIL
# ============================================================

def master_council(
    metrics,
    score,
    valuation
):

    growth = metrics[
        "earnings_growth"
    ]

    roe = metrics[
        "roe"
    ]

    pe = metrics[
        "pe"
    ]

    fcf = metrics[
        "fcf"
    ]

    result = {}

    # Buffett framework
    if score >= 85:

        if (
            valuation is not None
            and valuation["fair"] >= 1
            and valuation["fair"] >=
            valuation["fair"] * 0.85
        ):

            buffett = (
                "重点不是预测下一季度股价，而是判断这是不是一个能够长期产生大量现金的好生意。"
                "当前模型显示商业质量较强，下一步应该研究护城河、资本配置与长期竞争优势。"
            )

        else:

            buffett = (
                "商业质量值得研究，但价格仍然决定投资回报率。"
                "好公司并不意味着任何价格都值得买。"
            )

    elif score >= 70:

        buffett = (
            "有一定商业质量，但还没有足够证据证明它具备非常强的长期复利属性。"
        )

    else:

        buffett = (
            "目前没有足够证据把它定义为高确定性的长期复利资产。"
        )

    result[
        "Buffett"
    ] = buffett

    # Munger framework
    risks = []

    if (
        not np.isnan(pe)
        and pe > 40
    ):
        risks.append(
            "估值过高"
        )

    if (
        not np.isnan(growth)
        and growth < 0
    ):
        risks.append(
            "盈利下降"
        )

    if (
        not np.isnan(roe)
        and roe < 0.10
    ):
        risks.append(
            "资本回报偏低"
        )

    if (
        not np.isnan(fcf)
        and fcf <= 0
    ):
        risks.append(
            "现金流质量不足"
        )

    if not risks:

        risks.append(
            "永久性损失风险来自竞争格局、估值或管理层资本配置"
        )

    result[
        "Munger"
    ] = (
        "反向思考："
        + "、".join(risks)
        + "。"
    )

    # Duan Yongping framework
    if score >= 80:

        duan = (
            "先判断生意，再判断价格。"
            "如果生意足够好，短期价格波动不是核心问题；"
            "但如果价格远远透支未来增长，再好的生意也可能变成低回报投资。"
        )

    else:

        duan = (
            "不要因为便宜就自动认为值得买。"
            "首先需要证明这是一个值得长期持有的生意。"
        )

    result[
        "段永平"
    ] = duan

    # Lynch framework
    if (
        not np.isnan(growth)
        and not np.isnan(pe)
        and pe > 0
    ):

        growth_pct = (
            growth * 100
        )

        peg_like = (
            pe /
            max(
                growth_pct,
                1
            )
        )

        if peg_like < 1:

            lynch = (
                "增长相对于估值具有吸引力。"
            )

        elif peg_like < 2:

            lynch = (
                "增长与估值基本匹配。"
            )

        else:

            lynch = (
                "估值可能已经跑在增长之前。"
            )

    else:

        lynch = (
            "缺乏足够的增长与估值数据。"
        )

    result[
        "Lynch"
    ] = lynch

    # Fisher framework
    result[
        "Fisher"
    ] = (
        "重点观察长期市场空间、产品竞争力、研发能力、"
        "利润率趋势以及管理层执行能力。"
    )

    return result


# ============================================================
# BULL / BEAR ENGINE
# ============================================================

def bull_case(metrics, score):

    points = []

    growth = metrics[
        "revenue_growth"
    ]

    earnings = metrics[
        "earnings_growth"
    ]

    roe = metrics[
        "roe"
    ]

    fcf = metrics[
        "fcf"
    ]

    if (
        not np.isnan(growth)
        and growth > 0.10
    ):

        points.append(
            "收入仍保持正增长，为长期复利提供基础。"
        )

    if (
        not np.isnan(earnings)
        and earnings > 0.15
    ):

        points.append(
            "盈利增长明显，为估值扩张或业绩兑现提供支持。"
        )

    if (
        not np.isnan(roe)
        and roe > 0.20
    ):

        points.append(
            "资本回报率较高，说明企业使用股东资本的效率较好。"
        )

    if (
        not np.isnan(fcf)
        and fcf > 0
    ):

        points.append(
            "自由现金流为正，为股东回报和再投资提供基础。"
        )

    if not points:

        points.append(
            "当前数据没有形成特别强的多头证据。"
        )

    return points[:6]


def bear_case(metrics, technical):

    points = []

    growth = metrics[
        "revenue_growth"
    ]

    earnings = metrics[
        "earnings_growth"
    ]

    debt = metrics[
        "debt_equity"
    ]

    pe = metrics[
        "pe"
    ]

    fcf = metrics[
        "fcf"
    ]

    if (
        not np.isnan(growth)
        and growth < 0
    ):

        points.append(
            "收入出现负增长，说明商业扩张正在放缓。"
        )

    if (
        not np.isnan(earnings)
        and earnings < 0
    ):

        points.append(
            "盈利出现负增长，可能意味着经营或周期压力。"
        )

    if (
        not np.isnan(debt)
        and debt > 150
    ):

        points.append(
            "杠杆偏高，会放大经济周期和经营风险。"
        )

    if (
        not np.isnan(pe)
        and pe > 35
    ):

        points.append(
            "估值较高，增长不及预期时容易出现估值压缩。"
        )

    if (
        np.isnan(fcf)
        or fcf <= 0
    ):

        points.append(
            "自由现金流不足，使长期价值判断更困难。"
        )

    max_dd = technical.get(
        "max_drawdown",
        np.nan
    )

    if (
        not np.isnan(max_dd)
        and max_dd < -0.50
    ):

        points.append(
            "历史回撤较大，说明市场对该资产的风险定价可能非常激烈。"
        )

    points.append(
        "竞争者、技术变化、监管、利率或管理层资本配置都可能破坏当前投资逻辑。"
    )

    return points[:7]


# ============================================================
# DEVIL'S ADVOCATE
# ============================================================

def devil_advocate(
    metrics,
    score,
    valuation
):

    arguments = []

    pe = metrics[
        "pe"
    ]

    growth = metrics[
        "earnings_growth"
    ]

    revenue = metrics[
        "revenue_growth"
    ]

    debt = metrics[
        "debt_equity"
    ]

    fcf = metrics[
        "fcf"
    ]

    if score >= 80:

        arguments.append(
            "市场可能已经充分认识到公司的优秀，因此未来收益率未必与公司质量同样优秀。"
        )

    if (
        not np.isnan(pe)
        and pe > 35
    ):

        arguments.append(
            "高估值意味着任何业绩失误都可能同时受到盈利下修和估值压缩的双重打击。"
        )

    if (
        not np.isnan(growth)
        and growth < 0
    ):

        arguments.append(
            "盈利负增长可能意味着当前商业模式正在经历阶段性或结构性压力。"
        )

    if (
        not np.isnan(revenue)
        and revenue < 0.05
    ):

        arguments.append(
            "收入增长偏慢可能限制未来长期复利速度。"
        )

    if (
        not np.isnan(debt)
        and debt > 150
    ):

        arguments.append(
            "较高杠杆可能在经济下行期间放大经营风险。"
        )

    if (
        np.isnan(fcf)
        or fcf <= 0
    ):

        arguments.append(
            "如果利润无法转化为自由现金流，账面盈利的质量需要重新审视。"
        )

    arguments.extend(
        [
            "竞争对手可能通过价格、产品或技术改变竞争格局。",
            "利率变化可能改变市场愿意支付的估值倍数。",
            "管理层资本配置错误可能损害原本不错的商业模式。",
            "市场可能已经把最乐观的未来情景计入股价。"
        ]
    )

    return arguments[:8]


# ============================================================
# THESIS BREAKERS
# ============================================================

def thesis_breakers(metrics):

    breakers = []

    if (
        not np.isnan(
            metrics["revenue_growth"]
        )
    ):

        breakers.append(
            "收入增长连续恶化"
        )

    if (
        not np.isnan(
            metrics["earnings_growth"]
        )
    ):

        breakers.append(
            "盈利增长持续低于预期"
        )

    if (
        not np.isnan(
            metrics["profit_margin"]
        )
    ):

        breakers.append(
            "利润率出现结构性下降"
        )

    breakers.extend(
        [
            "自由现金流持续恶化",
            "核心产品竞争优势明显下降",
            "管理层资本配置出现重大问题",
            "行业竞争格局发生结构性变化",
            "估值远远跑在基本面前面"
        ]
    )

    return breakers


# ============================================================
# SCENARIO ENGINE
# ============================================================

def scenario_analysis(
    price,
    valuation
):

    if valuation is None:
        return None

    fair = valuation[
        "fair"
    ]

    return {

        "Bear":
            fair * 0.75,

        "Base":
            fair,

        "Bull":
            fair * 1.25
    }


# ============================================================
# POSITION PLAN
# ============================================================

def position_plan(
    price,
    valuation,
    risk_preference
):

    if valuation is None:

        return [
            (
                "观察仓",
                "0–10%",
                "估值证据不足。"
            )
        ]

    strong = valuation[
        "strong_buy"
    ]

    buy = valuation[
        "buy"
    ]

    fair = valuation[
        "fair"
    ]

    if price <= strong:

        if risk_preference == "保守":

            return [
                (
                    "第一笔",
                    "15%",
                    "先建立观察仓"
                ),
                (
                    "第二笔",
                    "10%",
                    "价格进一步确认"
                ),
                (
                    "第三笔",
                    "10%",
                    "基本面继续稳定"
                )
            ]

        if risk_preference == "进取":

            return [
                (
                    "第一笔",
                    "30%",
                    "安全边际较高"
                ),
                (
                    "第二笔",
                    "20%",
                    "继续确认"
                ),
                (
                    "第三笔",
                    "15%",
                    "基本面确认"
                )
            ]

        return [
            (
                "第一笔",
                "20%",
                "建立初始仓位"
            ),
            (
                "第二笔",
                "15%",
                "价格继续有吸引力"
            ),
            (
                "第三笔",
                "10%",
                "基本面确认"
            )
        ]

    if price <= buy:

        return [
            (
                "第一笔",
                "10–15%",
                "小仓位试探"
            ),
            (
                "第二笔",
                "10%",
                "价格进一步改善"
            ),
            (
                "第三笔",
                "5–10%",
                "确认基本面没有恶化"
            )
        ]

    if price <= fair:

        return [
            (
                "观察仓",
                "0–5%",
                "接近合理价值"
            ),
            (
                "等待",
                "现金",
                "等待更强安全边际"
            )
        ]

    return [
        (
            "不追涨",
            "0%",
            "当前价格缺乏安全边际"
        ),
        (
            "等待",
            "现金",
            "等待更好的价格"
        )
    ]


# ============================================================
# NEWS ENGINE
# ============================================================

def extract_news(news):

    rows = []

    if not isinstance(
        news,
        list
    ):
        return rows

    for item in news[:12]:

        try:

            content = item.get(
                "content",
                item
            )

            title = content.get(
                "title",
                ""
            )

            provider = content.get(
                "provider",
                {}
            )

            if isinstance(
                provider,
                dict
            ):

                publisher = provider.get(
                    "displayName",
                    ""
                )

            else:

                publisher = str(
                    provider
                )

            if title:

                rows.append(
                    {
                        "Title":
                            title,
                        "Publisher":
                            publisher
                    }
                )

        except Exception:
            continue

    return rows


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

    <div class="hero-subtitle">
    SIMON STOCK V7.0 · AI-NATIVE INVESTMENT RESEARCH ENGINE
    </div>

    <div class="hero-title">
    🧠 Simon Stock
    </div>

    <div class="hero-subtitle">
    Data × Quality × Growth × Valuation × Risk × Inversion
    </div>

    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🔎 Research")

    symbol = st.text_input(
        "股票代码",
        value="AAPL"
    ).upper().strip()

    period = st.selectbox(
        "历史数据",
        [
            "6mo",
            "1y",
            "2y",
            "5y",
            "10y"
        ],
        index=1
    )

    st.divider()

    st.header("🎯 Simon Settings")

    risk_preference = st.selectbox(
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
            "短期",
            "1–3年",
            "3–5年",
            "5年以上"
        ],
        index=2
    )

    st.divider()

    st.info(
        "核心分析无需 OpenAI API Key。"
    )

    st.caption(
        "Simon Stock V7.0"
    )

# ============================================================
# VALIDATION
# ============================================================

if not symbol:

    st.warning(
        "请输入股票代码。"
    )

    st.stop()

# ============================================================
# LOAD
# ============================================================

try:

    with st.spinner(
        "🧠 Simon 正在读取市场数据并构建研究模型..."
    ):

        history, info, news = load_stock(
            symbol,
            period
        )

except Exception as error:

    st.error(
        "读取股票数据失败。"
    )

    st.code(
        str(error)
    )

    st.stop()

if (
    history is None
    or history.empty
):

    st.error(
        f"没有找到 {symbol} 的有效市场数据。"
    )

    st.stop()

# ============================================================
# CALCULATIONS
# ============================================================

technical = technical_analysis(
    history
)

metrics = fundamental_metrics(
    info
)

quality = data_quality(
    history,
    info
)

price = technical.get(
    "price",
    np.nan
)

score, dimension_scores = calculate_simon_score(
    info,
    metrics,
    technical
)

valuation = fair_value_engine(
    info,
    price
)

price_score = price_attractiveness(
    price,
    valuation
)

verdict, verdict_reason = generate_verdict(
    score,
    price_score,
    price,
    valuation,
    risk_preference
)

masters = master_council(
    metrics,
    score,
    valuation
)

bull = bull_case(
    metrics,
    score
)

bear = bear_case(
    metrics,
    technical
)

devil = devil_advocate(
    metrics,
    score,
    valuation
)

breakers = thesis_breakers(
    metrics
)

scenarios = scenario_analysis(
    price,
    valuation
)

plan = position_plan(
    price,
    valuation,
    risk_preference
)

company = info.get(
    "longName",
    symbol
)

sector = info.get(
    "sector",
    "N/A"
)

industry = info.get(
    "industry",
    "N/A"
)

previous_price = np.nan

if len(history) >= 2:

    previous_price = safe_float(
        history["Close"].iloc[-2]
    )

daily_change = np.nan

if (
    not np.isnan(price)
    and not np.isnan(previous_price)
    and previous_price != 0
):

    daily_change = (
        price /
        previous_price -
        1
    )

# ============================================================
# TOP METRICS
# ============================================================

st.subheader(
    f"{company} · {symbol}"
)

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:

    st.metric(
        "Price",
        (
            f"${price:.2f}"
            if not np.isnan(price)
            else "N/A"
        ),
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

    pe = metrics[
        "pe"
    ]

    st.metric(
        "P/E",
        (
            f"{pe:.1f}x"
            if not np.isnan(pe)
            else "N/A"
        )
    )

with c4:

    st.metric(
        "ROE",
        percentage(
            metrics["roe"]
        )
    )

with c5:

    st.metric(
        "Revenue Growth",
        percentage(
            metrics[
                "revenue_growth"
            ]
        )
    )

with c6:

    st.metric(
        "Data Confidence",
        f"{quality['score']}/100"
    )

# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "🏠 Dashboard",
        "🧠 Simon Intelligence",
        "💰 Valuation",
        "🏆 Master Council",
        "⚔️ Bull vs Bear",
        "🧨 Devil",
        "📈 Technical",
        "⚔️ Battle",
        "💼 Portfolio",
        "📰 News"
    ]
)

# ============================================================
# DASHBOARD
# ============================================================

with tabs[0]:

    st.markdown(
        "## 🧠 Simon Final Verdict"
    )

    left, right = st.columns(
        [1, 2]
    )

    with left:

        st.markdown(
            f"""
            <div class="card">

            <div class="small">
            COMPANY QUALITY SCORE
            </div>

            <div class="score">
            {score}
            </div>

            <div>
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

            <div class="small">
            CURRENT DECISION
            </div>

            <div class="decision">
            {verdict}
            </div>

            <p>
            {verdict_reason}
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    # Quality / Price
    q1, q2, q3 = st.columns(3)

    with q1:

        st.metric(
            "Company Quality",
            f"{score}/100"
        )

    with q2:

        if np.isnan(price_score):

            st.metric(
                "Price Attractiveness",
                "N/A"
            )

        else:

            st.metric(
                "Price Attractiveness",
                f"{int(price_score)}/100"
            )

    with q3:

        st.metric(
            "Data Confidence",
            f"{quality['score']}/100"
        )

    st.markdown(
        "## 📊 Simon Dimensions"
    )

    dimension_cols = st.columns(
        len(dimension_scores)
    )

    for column, item in zip(
        dimension_cols,
        dimension_scores.items()
    ):

        name, value = item

        column.metric(
            name,
            f"{value}/100"
        )

    st.markdown(
        "## 📈 Price History"
    )

    chart_data = history[
        ["Close"]
    ].copy()

    chart_data.columns = [
        symbol
    ]

    st.line_chart(
        chart_data,
        height=400
    )

    if valuation is not None:

        st.markdown(
            "## 💰 Price vs Simon Fair Value"
        )

        difference = (
            price /
            valuation["fair"] -
            1
        )

        st.metric(
            "相对合理价值",
            f"{difference * 100:+.1f}%"
        )

    st.markdown(
        "## 🏢 Company"
    )

    st.caption(
        f"Sector: {sector} · Industry: {industry}"
    )

    company_description = info.get(
        "longBusinessSummary",
        "暂无公司简介。"
    )

    st.write(
        company_description
    )

# ============================================================
# SIMON INTELLIGENCE
# ============================================================

with tabs[1]:

    st.markdown(
        "## 🧠 Simon Intelligence Engine"
    )

    st.info(
        "Simon 不只回答“公司好不好”，而是同时回答：公司质量、价格、安全边际，以及我可能错在哪里。"
    )

    st.markdown(
        "### 📡 Data Quality"
    )

    d1, d2 = st.columns(2)

    with d1:

        st.metric(
            "Data Score",
            f"{quality['score']}/100"
        )

    with d2:

        st.metric(
            "Confidence",
            quality["confidence"]
        )

    if quality["checks"]:

        st.success(
            "✓ " +
            " · ".join(
                quality["checks"]
            )
        )

    if quality["issues"]:

        for issue in quality["issues"]:

            st.warning(
                "⚠️ " + issue
            )

    st.divider()

    st.markdown(
        "### 📊 Core Evidence"
    )

    evidence = pd.DataFrame(
        {
            "指标": [
                "Revenue Growth",
                "Earnings Growth",
                "Profit Margin",
                "ROE",
                "Debt / Equity",
                "Free Cash Flow",
                "FCF Yield",
                "P/E",
                "Forward P/E"
            ],

            "数值": [
                percentage(
                    metrics["revenue_growth"]
                ),

                percentage(
                    metrics["earnings_growth"]
                ),

                percentage(
                    metrics["profit_margin"]
                ),

                percentage(
                    metrics["roe"]
                ),

                (
                    f"{metrics['debt_equity']:.1f}"
                    if not np.isnan(
                        metrics["debt_equity"]
                    )
                    else "N/A"
                ),

                dollar(
                    metrics["fcf"]
                ),

                percentage(
                    metrics["free_cashflow_yield"]
                ),

                (
                    f"{metrics['pe']:.1f}x"
                    if not np.isnan(
                        metrics["pe"]
                    )
                    else "N/A"
                ),

                (
                    f"{metrics['forward_pe']:.1f}x"
                    if not np.isnan(
                        metrics["forward_pe"]
                    )
                    else "N/A"
                )
            ]
        }
    )

    st.dataframe(
        evidence,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        "### 🎯 Simon Three Questions"
    )

    questions = [
        "这是好生意吗？",
        "这个价格值得买吗？",
        "如果我错了，最可能错在哪里？"
    ]

    for question in questions:

        st.markdown(
            f"☐ **{question}**"
        )

    st.divider()

    st.markdown(
        "### 🧩 Simon Principles"
    )

    principles = [
        "好公司 ≠ 好股票",
        "价格决定未来回报率",
        "安全边际比预测更重要",
        "先寻找反方证据",
        "现金也是选择权",
        "数据不足时降低置信度"
    ]

    for principle in principles:

        st.markdown(
            f"**• {principle}**"
        )

# ============================================================
# VALUATION
# ============================================================

with tabs[2]:

    st.markdown(
        "## 💰 Simon Valuation Lab"
    )

    if valuation is None:

        st.warning(
            "当前公司缺少足够的现金流或估值数据，暂时无法形成可靠的综合估值。"
        )

    else:

        v1, v2, v3, v4, v5 = st.columns(5)

        with v1:

            st.metric(
                "🔥 Strong Buy",
                f"${valuation['strong_buy']:.2f}"
            )

        with v2:

            st.metric(
                "🟢 Buy Zone",
                f"${valuation['buy']:.2f}"
            )

        with v3:

            st.metric(
                "🟡 Fair Value",
                f"${valuation['fair']:.2f}"
            )

        with v4:

            st.metric(
                "🟠 Expensive",
                f"${valuation['expensive']:.2f}"
            )

        with v5:

            st.metric(
                "🔴 Danger",
                f"${valuation['danger']:.2f}"
            )

        st.divider()

        st.markdown(
            "### 📊 Valuation Components"
        )

        component_rows = []

        dcf = valuation.get(
            "dcf"
        )

        if dcf is not None:

            component_rows.append(
                {
                    "模型":
                        "DCF Base",
                    "估值":
                        dcf.get(
                            "Base",
                            np.nan
                        )
                }
            )

        component_rows.append(
            {
                "模型":
                    "Multiple",
                "估值":
                    valuation.get(
                        "multiple",
                        np.nan
                    )
            }
        )

        component_rows.append(
            {
                "模型":
                    "FCF Yield",
                "估值":
                    valuation.get(
                        "fcf_value",
                        np.nan
                    )
            }
        )

        component_df = pd.DataFrame(
            component_rows
        )

        component_df[
            "估值"
        ] = component_df[
            "估值"
        ].apply(
            lambda x:
            round(x, 2)
            if not np.isnan(
                safe_float(x)
            )
            else np.nan
        )

        st.dataframe(
            component_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            "### 📉 Discount / Premium"
        )

        difference = (
            price /
            valuation["fair"] -
            1
        )

        st.metric(
            "当前价格相对合理价值",
            f"{difference * 100:+.1f}%"
        )

        if difference <= -0.20:

            st.success(
                "价格明显低于模型合理价值。"
            )

        elif difference <= 0:

            st.info(
                "价格不高于模型合理价值。"
            )

        elif difference <= 0.20:

            st.warning(
                "价格高于模型合理价值。"
            )

        else:

            st.error(
                "价格明显高于模型合理价值。"
            )

        if dcf is not None:

            st.markdown(
                "### 📊 DCF Scenarios"
            )

            dcf_table = pd.DataFrame(
                {
                    "Scenario": [
                        "Bear",
                        "Base",
                        "Bull"
                    ],

                    "Fair Value": [
                        dcf["Bear"],
                        dcf["Base"],
                        dcf["Bull"]
                    ]
                }
            )

            dcf_table[
                "Upside / Downside"
            ] = (
                (
                    dcf_table[
                        "Fair Value"
                    ] /
                    price -
                    1
                ) * 100
            ).round(1)

            dcf_table[
                "Upside / Downside"
            ] = (
                dcf_table[
                    "Upside / Downside"
                ].astype(str)
                + "%"
            )

            st.dataframe(
                dcf_table,
                use_container_width=True,
                hide_index=True
            )

        st.markdown(
            "### 🎯 Simon Position Plan"
        )

        plan_df = pd.DataFrame(
            plan,
            columns=[
                "动作",
                "资金比例",
                "逻辑"
            ]
        )

        st.dataframe(
            plan_df,
            use_container_width=True,
            hide_index=True
        )

# ============================================================
# MASTER COUNCIL
# ============================================================

with tabs[3]:

    st.markdown(
        "## 🏆 Investment Master Council"
    )

    st.caption(
        "这里不是声称这些投资大师会对具体股票作出这样的判断，而是将他们公开、广为人知的投资原则转化为检查框架。"
    )

    for master, opinion in masters.items():

        with st.expander(
            f"🏆 {master}",
            expanded=True
        ):

            st.write(
                opinion
            )

    st.divider()

    st.markdown(
        "### 🧠 Five Master Questions"
    )

    master_questions = [
        "如果市场关闭五年，我还愿意持有它吗？",
        "如果不能卖出，我还愿意买入吗？",
        "这家公司未来十年还能变得更强吗？",
        "管理层能否合理配置资本？",
        "现在的价格是否提供足够安全边际？"
    ]

    for index, question in enumerate(
        master_questions,
        1
    ):

        st.markdown(
            f"**{index}.** {question}"
        )

# ============================================================
# BULL VS BEAR
# ============================================================

with tabs[4]:

    st.markdown(
        "## ⚔️ Bull Case vs Bear Case"
    )

    bull_col, bear_col = st.columns(2)

    with bull_col:

        st.success(
            "🟢 BULL CASE"
        )

        for point in bull:

            st.markdown(
                f"• {point}"
            )

    with bear_col:

        st.error(
            "🔴 BEAR CASE"
        )

        for point in bear:

            st.markdown(
                f"• {point}"
            )

    st.divider()

    st.markdown(
        "## 🎯 What Would Change The Thesis?"
    )

    for item in breakers:

        st.markdown(
            f"☐ {item}"
        )

# ============================================================
# DEVIL
# ============================================================

with tabs[5]:

    st.markdown(
        "## 🧨 Devil's Advocate"
    )

    st.error(
        "这个模块故意站在“不买 / 卖出”的角度攻击当前投资逻辑。"
    )

    for index, argument in enumerate(
        devil,
        1
    ):

        st.markdown(
            f"### {index}. {argument}"
        )

    st.divider()

    st.markdown(
        "### 🔥 Simon's Hard Questions"
    )

    hard_questions = [
        "如果未来三年增长只有预期的一半，现在的估值还合理吗？",
        "如果竞争对手突然变强，公司还有护城河吗？",
        "如果利率长期维持高位，估值还能撑住吗？",
        "如果我今天没有持仓，我还会买入吗？",
        "如果股价下跌30%，我的投资逻辑会改变吗？"
    ]

    for question in hard_questions:

        st.markdown(
            f"**• {question}**"
        )

# ============================================================
# TECHNICAL
# ============================================================

with tabs[6]:

    st.markdown(
        "## 📈 Technical Intelligence"
    )

    t1, t2, t3, t4, t5 = st.columns(5)

    with t1:

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

    with t2:

        sma20 = technical.get(
            "sma20",
            np.nan
        )

        st.metric(
            "SMA 20",
            (
                f"${sma20:.2f}"
                if not np.isnan(sma20)
                else "N/A"
            )
        )

    with t3:

        sma50 = technical.get(
            "sma50",
            np.nan
        )

        st.metric(
            "SMA 50",
            (
                f"${sma50:.2f}"
                if not np.isnan(sma50)
                else "N/A"
            )
        )

    with t4:

        sma200 = technical.get(
            "sma200",
            np.nan
        )

        st.metric(
            "SMA 200",
            (
                f"${sma200:.2f}"
                if not np.isnan(sma200)
                else "N/A"
            )
        )

    with t5:

        volatility = technical.get(
            "volatility20",
            np.nan
        )

        st.metric(
            "Volatility",
            (
                f"{volatility * 100:.1f}%"
                if not np.isnan(volatility)
                else "N/A"
            )
        )

    st.markdown(
        "### 📊 Trend"
    )

    sma20 = technical.get(
        "sma20",
        np.nan
    )

    sma50 = technical.get(
        "sma50",
        np.nan
    )

    sma200 = technical.get(
        "sma200",
        np.nan
    )

    if (
        not np.isnan(sma20)
        and not np.isnan(sma50)
        and not np.isnan(sma200)
    ):

        if (
            price > sma20
            and sma20 > sma50
            and sma50 > sma200
        ):

            st.success(
                "🟢 多周期趋势偏强。"
            )

        elif (
            price < sma20
            and sma20 < sma50
            and sma50 < sma200
        ):

            st.error(
                "🔴 多周期趋势偏弱。"
            )

        else:

            st.warning(
                "🟡 趋势处于混合状态。"
            )

    st.markdown(
        "### 📉 Drawdown & Range"
    )

    dd1, dd2, dd3 = st.columns(3)

    with dd1:

        max_dd = technical.get(
            "max_drawdown",
            np.nan
        )

        st.metric(
            "Max Drawdown",
            (
                f"{max_dd * 100:.1f}%"
                if not np.isnan(max_dd)
                else "N/A"
            )
        )

    with dd2:

        high52 = technical.get(
            "52w_high",
            np.nan
        )

        st.metric(
            "52W High",
            (
                f"${high52:.2f}"
                if not np.isnan(high52)
                else "N/A"
            )
        )

    with dd3:

        low52 = technical.get(
            "52w_low",
            np.nan
        )

        st.metric(
            "52W Low",
            (
                f"${low52:.2f}"
                if not np.isnan(low52)
                else "N/A"
            )
        )

    st.line_chart(
        history["Close"],
        height=400
    )

# ============================================================
# BATTLE
# ============================================================

with tabs[7]:

    st.markdown(
        "## ⚔️ Simon Stock Battle"
    )

    battle_input = st.text_input(
        "输入 2–5 个股票代码，用逗号分隔",
        value="AAPL,GOOGL,AVGO"
    )

    if st.button(
        "⚔️ Start Battle",
        type="primary"
    ):

        battle_symbols = [
            item.strip().upper()
            for item in
            battle_input.split(",")
            if item.strip()
        ]

        battle_symbols = list(
            dict.fromkeys(
                battle_symbols
            )
        )[:5]

        battle_rows = []

        progress = st.progress(
            0
        )

        total = len(
            battle_symbols
        )

        for index, battle_symbol in enumerate(
            battle_symbols
        ):

            try:

                h, inf, _ = load_stock(
                    battle_symbol,
                    "1y"
                )

                if h.empty:
                    continue

                tech = technical_analysis(
                    h
                )

                met = fundamental_metrics(
                    inf
                )

                sc, dims = calculate_simon_score(
                    inf,
                    met,
                    tech
                )

                p = tech.get(
                    "price",
                    np.nan
                )

                va = fair_value_engine(
                    inf,
                    p
                )

                fair_price = np.nan
                price_score_b = np.nan

                if va is not None:

                    fair_price = va[
                        "fair"
                    ]

                    price_score_b = (
                        price_attractiveness(
                            p,
                            va
                        )
                    )

                battle_rows.append(
                    {
                        "Ticker":
                            battle_symbol,

                        "Price":
                            round(
                                p,
                                2
                            ),

                        "Quality":
                            sc,

                        "Price Score":
                            (
                                int(
                                    price_score_b
                                )
                                if not np.isnan(
                                    price_score_b
                                )
                                else np.nan
                            ),

                        "Fair Value":
                            (
                                round(
                                    fair_price,
                                    2
                                )
                                if not np.isnan(
                                    fair_price
                                )
                                else np.nan
                            ),

                        "Upside":
                            (
                                f"{(
                                    fair_price /
                                    p -
                                    1
                                ) * 100:+.1f}%"
                                if (
                                    not np.isnan(
                                        fair_price
                                    )
                                    and p > 0
                                )
                                else "N/A"
                            )
                    }
                )

            except Exception:
                pass

            progress.progress(
                (index + 1) /
                max(
                    total,
                    1
                )
            )

        if battle_rows:

            battle_df = pd.DataFrame(
                battle_rows
            )

            battle_df[
                "Battle Score"
            ] = (
                battle_df[
                    "Quality"
                ] * 0.65
                +
                battle_df[
                    "Price Score"
                ].fillna(50) * 0.35
            ).round(0).astype(int)

            battle_df = battle_df.sort_values(
                "Battle Score",
                ascending=False
            )

            st.dataframe(
                battle_df,
                use_container_width=True,
                hide_index=True
            )

            winner = battle_df.iloc[0]

            st.success(
                f"🏆 Simon Winner："
                f"{winner['Ticker']} · "
                f"Battle Score "
                f"{winner['Battle Score']}/100"
            )

        else:

            st.error(
                "没有成功读取 Battle 数据。"
            )

# ============================================================
# PORTFOLIO
# ============================================================

with tabs[8]:

    st.markdown(
        "## 💼 Simon Portfolio Brain"
    )

    st.write(
        "格式：股票代码,股数,平均成本"
    )

    portfolio_text = st.text_area(
        "你的持仓",
        value=(
            "AAPL,2,310\n"
            "GOOGL,2,342\n"
            "AVGO,2,352"
        ),
        height=160
    )

    if st.button(
        "🧠 Analyze Portfolio",
        type="primary"
    ):

        portfolio_rows = []

        lines = (
            portfolio_text
            .splitlines()
        )

        for line in lines:

            parts = [
                part.strip()
                for part in
                line.split(",")
            ]

            if len(parts) < 3:
                continue

            try:

                portfolio_symbol = (
                    parts[0].upper()
                )

                shares = float(
                    parts[1]
                )

                cost = float(
                    parts[2]
                )

                h, inf, _ = load_stock(
                    portfolio_symbol,
                    "5d"
                )

                if h.empty:
                    continue

                current_price = safe_float(
                    h["Close"].iloc[-1]
                )

                market_value = (
                    current_price *
                    shares
                )

                invested = (
                    cost *
                    shares
                )

                pnl = (
                    market_value -
                    invested
                )

                pnl_pct = safe_div(
                    pnl,
                    invested,
                    0
                )

                portfolio_rows.append(
                    {
                        "Ticker":
                            portfolio_symbol,

                        "Shares":
                            shares,

                        "Cost":
                            round(
                                cost,
                                2
                            ),

                        "Price":
                            round(
                                current_price,
                                2
                            ),

                        "Value":
                            round(
                                market_value,
                                2
                            ),

                        "P/L":
                            round(
                                pnl,
                                2
                            ),

                        "P/L %":
                            f"{pnl_pct * 100:+.2f}%"
                    }
                )

            except Exception:
                continue

        if portfolio_rows:

            portfolio_df = pd.DataFrame(
                portfolio_rows
            )

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

            p1, p2, p3 = st.columns(3)

            p1.metric(
                "Portfolio Value",
                f"${total_value:,.2f}"
            )

            p2.metric(
                "Total P/L",
                f"${total_pnl:+,.2f}"
            )

            p3.metric(
                "Positions",
                len(portfolio_df)
            )

            # Concentration
            portfolio_df[
                "Weight"
            ] = (
                portfolio_df[
                    "Value"
                ] /
                total_value
                if total_value > 0
                else 0
            )

            largest_weight = (
                portfolio_df[
                    "Weight"
                ].max()
            )

            st.metric(
                "Largest Position",
                f"{largest_weight * 100:.1f}%"
            )

            if largest_weight >= 0.50:

                st.error(
                    "⚠️ 单一持仓占比超过50%，组合集中度很高。"
                )

            elif largest_weight >= 0.30:

                st.warning(
                    "⚠️ 最大持仓超过30%，需要关注集中风险。"
                )

            else:

                st.success(
                    "组合集中度目前没有明显异常。"
                )

        else:

            st.error(
                "没有读取到有效持仓。"
            )

# ============================================================
# NEWS
# ============================================================

with tabs[9]:

    st.markdown(
        "## 📰 Latest News"
    )

    news_rows = extract_news(
        news
    )

    if news_rows:

        for item in news_rows:

            st.markdown(
                f"### {item['Title']}"
            )

            if item[
                "Publisher"
            ]:

                st.caption(
                    item[
                        "Publisher"
                    ]
                )

            st.divider()

    else:

        st.info(
            "当前无法从 Yahoo Finance 获取新闻。"
        )

# ============================================================
# FINAL SUMMARY
# ============================================================

st.divider()

st.markdown(
    "## 🧠 Simon Final Summary"
)

summary_left, summary_right = st.columns(
    2
)

with summary_left:

    st.markdown(
        f"""
        <div class="card">

        <div class="small">
        COMPANY QUALITY
        </div>

        <h2>{score}/100</h2>

        <p>
        Business Quality：
        {dimension_scores["Business Quality"]}/100
        </p>

        <p>
        Growth：
        {dimension_scores["Growth"]}/100
        </p>

        <p>
        Profitability：
        {dimension_scores["Profitability"]}/100
        </p>

        <p>
        Financial Strength：
        {dimension_scores["Financial Strength"]}/100
        </p>

        <p>
        Capital Efficiency：
        {dimension_scores["Capital Efficiency"]}/100
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

with summary_right:

    fair_text = "N/A"

    if valuation is not None:

        fair_text = (
            f"${valuation['fair']:.2f}"
        )

    price_score_text = (
        "N/A"
        if np.isnan(
            price_score
        )
        else f"{int(price_score)}/100"
    )

    st.markdown(
        f"""
        <div class="card">

        <div class="small">
        PRICE DISCIPLINE
        </div>

        <h2>{verdict}</h2>

        <p>
        Current Price：
        ${price:.2f}
        </p>

        <p>
        Simon Fair Value：
        {fair_text}
        </p>

        <p>
        Price Attractiveness：
        {price_score_text}
        </p>

        <p>
        Investment Horizon：
        {horizon}
        </p>

        <p>
        Data Confidence：
        {quality["score"]}/100
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

st.caption(
    "Simon Stock V7.0 · Data powered by Yahoo Finance / yfinance"
)

st.caption(
    "For investment research and education only. "
    "Not financial advice. No investment return is guaranteed."
)

st.caption(
    "Analysis time: "
    +
    datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
)