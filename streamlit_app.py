import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================
# SIMON STOCK V6.0
# Ultimate Free Intelligence
# No OpenAI API required
# ============================================================

st.set_page_config(
    page_title="Simon Stock V6.0",
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
        opacity: 0.65;
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
        opacity: 0.65;
        font-size: 13px;
    }

    .green {
        font-weight: 800;
    }

    .yellow {
        font-weight: 800;
    }

    .red {
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=np.nan):
    try:
        result = float(value)
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def percentage(value):
    value = safe_float(value)
    if np.isnan(value):
        return "N/A"
    return f"{value * 100:.1f}%"


def dollar(value):
    value = safe_float(value)

    if np.isnan(value):
        return "N/A"

    if abs(value) >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"

    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"

    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"

    return f"${value:,.2f}"


def clamp(value, low=0, high=100):
    return max(low, min(high, value))


def get_value(info, *keys):
    for key in keys:
        if key in info:
            value = safe_float(info.get(key))
            if not np.isnan(value):
                return value
    return np.nan


# ============================================================
# DATA
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
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
# TECHNICAL ANALYSIS
# ============================================================

def technical_analysis(history):
    result = {}

    if history is None or history.empty:
        return result

    close = history["Close"].dropna()

    if len(close) == 0:
        return result

    price = safe_float(close.iloc[-1])
    result["price"] = price

    if len(close) >= 20:
        result["sma20"] = safe_float(
            close.rolling(20).mean().iloc[-1]
        )

    if len(close) >= 50:
        result["sma50"] = safe_float(
            close.rolling(50).mean().iloc[-1]
        )

    if len(close) >= 200:
        result["sma200"] = safe_float(
            close.rolling(200).mean().iloc[-1]
        )

    if len(close) >= 15:
        delta = close.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)

        rsi = 100 - (100 / (1 + rs))

        result["rsi"] = safe_float(rsi.iloc[-1])

    if len(close) >= 30:
        result["return30"] = (
            safe_float(close.iloc[-1]) /
            safe_float(close.iloc[-30])
            - 1
        )

    if len(close) >= 252:
        result["return1y"] = (
            safe_float(close.iloc[-1]) /
            safe_float(close.iloc[-252])
            - 1
        )

    rolling_high = close.cummax()

    drawdown = (
        close / rolling_high - 1
    )

    result["max_drawdown"] = safe_float(
        drawdown.min()
    )

    return result


# ============================================================
# FUNDAMENTAL METRICS
# ============================================================

def fundamental_metrics(info):
    return {
        "revenue_growth": get_value(
            info,
            "revenueGrowth"
        ),
        "earnings_growth": get_value(
            info,
            "earningsGrowth"
        ),
        "profit_margin": get_value(
            info,
            "profitMargins"
        ),
        "operating_margin": get_value(
            info,
            "operatingMargins"
        ),
        "roe": get_value(
            info,
            "returnOnEquity"
        ),
        "roa": get_value(
            info,
            "returnOnAssets"
        ),
        "debt_equity": get_value(
            info,
            "debtToEquity"
        ),
        "current_ratio": get_value(
            info,
            "currentRatio"
        ),
        "fcf": get_value(
            info,
            "freeCashflow"
        ),
        "pe": get_value(
            info,
            "trailingPE"
        ),
        "forward_pe": get_value(
            info,
            "forwardPE"
        ),
        "peg": get_value(
            info,
            "pegRatio"
        ),
        "price_to_sales": get_value(
            info,
            "priceToSalesTrailing12Months"
        ),
        "price_to_book": get_value(
            info,
            "priceToBook"
        ),
        "beta": get_value(
            info,
            "beta"
        ),
        "market_cap": get_value(
            info,
            "marketCap"
        )
    }


# ============================================================
# SCORE COMPONENT
# ============================================================

def score_growth(metrics):
    score = 50

    growth = metrics["revenue_growth"]
    earnings = metrics["earnings_growth"]

    if not np.isnan(growth):
        if growth >= 0.30:
            score += 35
        elif growth >= 0.20:
            score += 25
        elif growth >= 0.10:
            score += 12
        elif growth >= 0:
            score -= 2
        else:
            score -= 25

    if not np.isnan(earnings):
        if earnings >= 0.30:
            score += 30
        elif earnings >= 0.20:
            score += 20
        elif earnings >= 0.10:
            score += 10
        elif earnings < 0:
            score -= 20

    return clamp(score)


def score_profitability(metrics):
    score = 50

    roe = metrics["roe"]
    margin = metrics["profit_margin"]
    operating = metrics["operating_margin"]

    if not np.isnan(roe):
        if roe >= 0.30:
            score += 35
        elif roe >= 0.20:
            score += 25
        elif roe >= 0.10:
            score += 10
        elif roe < 0:
            score -= 25

    if not np.isnan(margin):
        if margin >= 0.25:
            score += 25
        elif margin >= 0.15:
            score += 15
        elif margin < 0:
            score -= 25

    if not np.isnan(operating):
        if operating >= 0.25:
            score += 20
        elif operating >= 0.15:
            score += 10
        elif operating < 0:
            score -= 15

    return clamp(score)


def score_financial(metrics):
    score = 50

    debt = metrics["debt_equity"]
    current = metrics["current_ratio"]
    fcf = metrics["fcf"]

    if not np.isnan(debt):
        if debt < 40:
            score += 25
        elif debt < 80:
            score += 15
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
            score += 20
        else:
            score -= 20

    return clamp(score)


def score_business(metrics, info):
    score = 50

    market_cap = metrics["market_cap"]

    if not np.isnan(market_cap):
        if market_cap >= 500_000_000_000:
            score += 30
        elif market_cap >= 100_000_000_000:
            score += 22
        elif market_cap >= 10_000_000_000:
            score += 12
        elif market_cap >= 1_000_000_000:
            score += 5

    sector = str(
        info.get("sector", "")
    ).lower()

    industry = str(
        info.get("industry", "")
    ).lower()

    business_text = sector + " " + industry

    if "technology" in business_text:
        score += 8

    if "healthcare" in business_text:
        score += 5

    if "consumer" in business_text:
        score += 5

    return clamp(score)


# ============================================================
# VALUATION SCORE
# ============================================================

def score_valuation(metrics):
    score = 50

    pe = metrics["pe"]
    forward_pe = metrics["forward_pe"]
    peg = metrics["peg"]

    chosen_pe = forward_pe

    if np.isnan(chosen_pe):
        chosen_pe = pe

    if not np.isnan(chosen_pe) and chosen_pe > 0:

        if chosen_pe < 15:
            score += 35

        elif chosen_pe < 20:
            score += 25

        elif chosen_pe < 25:
            score += 15

        elif chosen_pe < 35:
            score += 5

        elif chosen_pe < 50:
            score -= 15

        else:
            score -= 30

    if not np.isnan(peg) and peg > 0:

        if peg < 1:
            score += 20

        elif peg < 1.5:
            score += 10

        elif peg > 3:
            score -= 20

    return clamp(score)


# ============================================================
# RISK SCORE
# ============================================================

def score_risk(metrics, technical):
    score = 70

    beta = metrics["beta"]
    debt = metrics["debt_equity"]

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

    drawdown = technical.get(
        "max_drawdown",
        np.nan
    )

    if not np.isnan(drawdown):

        if drawdown > -0.20:
            score += 10

        elif drawdown > -0.40:
            score += 0

        elif drawdown > -0.60:
            score -= 10

        else:
            score -= 20

    return clamp(score)


# ============================================================
# SIMON SCORE
# ============================================================

def calculate_simon_score(metrics, info, technical):
    business = score_business(
        metrics,
        info
    )

    growth = score_growth(
        metrics
    )

    profitability = score_profitability(
        metrics
    )

    financial = score_financial(
        metrics
    )

    valuation = score_valuation(
        metrics
    )

    risk = score_risk(
        metrics,
        technical
    )

    scores = {
        "Business Quality": business,
        "Growth": growth,
        "Profitability": profitability,
        "Financial Strength": financial,
        "Valuation": valuation,
        "Risk": risk
    }

    weights = {
        "Business Quality": 0.20,
        "Growth": 0.18,
        "Profitability": 0.18,
        "Financial Strength": 0.16,
        "Valuation": 0.18,
        "Risk": 0.10
    }

    total = 0

    for key in scores:
        total += (
            scores[key] *
            weights[key]
        )

    return int(round(total)), scores


# ============================================================
# DCF
# ============================================================

def dcf_model(info, price):
    fcf = safe_float(
        info.get("freeCashflow")
    )

    shares = safe_float(
        info.get("sharesOutstanding")
    )

    if (
        np.isnan(fcf)
        or fcf <= 0
        or np.isnan(shares)
        or shares <= 0
    ):
        return None

    fcf_per_share = (
        fcf / shares
    )

    growth = safe_float(
        info.get("earningsGrowth")
    )

    if np.isnan(growth):
        growth = 0.10

    growth = max(
        -0.05,
        min(
            0.25,
            growth
        )
    )

    scenarios = {}

    assumptions = {
        "Bear": (
            max(-0.02, growth - 0.08),
            0.09
        ),
        "Base": (
            max(0.00, growth - 0.02),
            0.085
        ),
        "Bull": (
            min(0.25, growth + 0.04),
            0.08
        )
    }

    for name, values in assumptions.items():

        growth_rate = values[0]
        discount = values[1]

        projected = fcf_per_share
        present_value = 0

        for year in range(1, 6):

            projected *= (
                1 + growth_rate
            )

            present_value += (
                projected /
                ((1 + discount) ** year)
            )

        terminal_growth = min(
            0.035,
            max(
                0.015,
                growth_rate
            )
        )

        terminal = (
            projected *
            (1 + terminal_growth)
            /
            (discount - terminal_growth)
        )

        terminal_pv = (
            terminal /
            ((1 + discount) ** 5)
        )

        fair_value = (
            present_value +
            terminal_pv
        )

        scenarios[name] = max(
            0,
            fair_value
        )

    return scenarios


# ============================================================
# MULTIPLE VALUATION
# ============================================================

def multiple_fair_value(info, price):
    pe = safe_float(
        info.get("trailingPE")
    )

    forward_pe = safe_float(
        info.get("forwardPE")
    )

    earnings_growth = safe_float(
        info.get("earningsGrowth")
    )

    values = []

    if not np.isnan(pe) and pe > 0:

        target_pe = 22

        if not np.isnan(earnings_growth):

            if earnings_growth >= 0.25:
                target_pe = 30

            elif earnings_growth >= 0.15:
                target_pe = 27

            elif earnings_growth >= 0.08:
                target_pe = 24

            elif earnings_growth < 0:
                target_pe = 18

        values.append(
            price *
            target_pe /
            pe
        )

    if not np.isnan(forward_pe) and forward_pe > 0:

        values.append(
            price *
            24 /
            forward_pe
        )

    if not values:
        return np.nan

    return float(
        np.median(values)
    )


# ============================================================
# FAIR VALUE ENGINE
# ============================================================

def fair_value_engine(info, price):
    dcf = dcf_model(
        info,
        price
    )

    multiple = multiple_fair_value(
        info,
        price
    )

    estimates = []

    if dcf is not None:

        if not np.isnan(
            dcf.get("Base", np.nan)
        ):
            estimates.append(
                dcf["Base"]
            )

    if not np.isnan(multiple):
        estimates.append(
            multiple
        )

    if not estimates:
        return None

    fair = float(
        np.median(estimates)
    )

    fair = max(
        price * 0.50,
        min(
            price * 2.00,
            fair
        )
    )

    result = {
        "fair": fair,
        "strong_buy": fair * 0.68,
        "buy": fair * 0.82,
        "expensive": fair * 1.18,
        "danger": fair * 1.40
    }

    if dcf is not None:
        result["dcf"] = dcf

    result["multiple"] = multiple

    return result


# ============================================================
# VERDICT
# ============================================================

def generate_verdict(
    score,
    price,
    valuation
):
    if valuation is None:
        if score >= 85:
            return (
                "🟡 GREAT COMPANY",
                "公司质量较高，但目前缺乏可靠估值数据。"
            )

        if score >= 70:
            return (
                "🟡 WATCH",
                "基本面尚可，等待更好的价格。"
            )

        return (
            "🔴 AVOID",
            "当前基本面证据不足。"
        )

    strong_buy = valuation["strong_buy"]
    buy = valuation["buy"]
    fair = valuation["fair"]
    expensive = valuation["expensive"]
    danger = valuation["danger"]

    if price <= strong_buy:

        if score >= 80:
            return (
                "🟢 STRONG BUY",
                "高质量 + 极具安全边际。"
            )

        return (
            "🟢 VALUE OPPORTUNITY",
            "价格具有吸引力，但仍需关注基本面。"
        )

    if price <= buy:

        if score >= 75:
            return (
                "🟢 BUY",
                "公司质量与价格目前较为匹配。"
            )

        return (
            "🟡 SMALL POSITION",
            "可以观察或小仓位试探。"
        )

    if price <= fair:

        if score >= 85:
            return (
                "🟡 GREAT COMPANY / FAIR PRICE",
                "公司很好，但没有明显安全边际。"
            )

        return (
            "🟡 WATCH",
            "合理价格，耐心比追涨更重要。"
        )

    if price <= expensive:

        return (
            "🟠 EXPENSIVE",
            "好公司可能仍然是好公司，但价格开始影响未来回报。"
        )

    if price <= danger:

        return (
            "🔴 HIGH RISK",
            "估值明显偏高，安全边际不足。"
        )

    return (
        "🔴 AVOID / WAIT",
        "当前价格远高于模型合理价值。"
    )


# ============================================================
# MASTER COUNCIL
# ============================================================

def master_council(
    info,
    metrics,
    score,
    price,
    valuation
):
    result = {}

    growth = metrics["earnings_growth"]
    roe = metrics["roe"]
    pe = metrics["pe"]

    # Buffett
    if score >= 82:

        if (
            valuation is not None
            and price <= valuation["fair"]
        ):
            buffett = (
                "商业质量和价格目前较协调。"
                "核心问题是未来十年的竞争优势能否继续保持。"
            )
        else:
            buffett = (
                "这是可能值得长期研究的好生意，"
                "但当前价格是最大的变量。"
            )

    elif score >= 70:

        buffett = (
            "有一定商业质量，但还不足以仅凭品牌或故事长期持有。"
        )

    else:

        buffett = (
            "目前证据不足以把它视为高确定性的长期复利资产。"
        )

    result["Buffett"] = buffett

    # Munger
    munger_risks = []

    if not np.isnan(pe) and pe > 40:
        munger_risks.append(
            "估值过高"
        )

    if not np.isnan(growth) and growth < 0:
        munger_risks.append(
            "盈利下降"
        )

    if not np.isnan(roe) and roe < 0.10:
        munger_risks.append(
            "资本回报不足"
        )

    if not munger_risks:
        munger_risks.append(
            "竞争优势被削弱"
        )

    result["Munger"] = (
        "反向思考：最大的永久性损失风险可能来自 "
        + "、".join(munger_risks)
        + "。"
    )

    # Duan Yongping
    if score >= 80:

        if (
            valuation is not None
            and price <= valuation["buy"]
        ):
            duan = (
                "生意质量、管理层和价格目前较协调，"
                "符合长期投资需要的基本框架。"
            )

        else:
            duan = (
                "生意可能很好，但价格不够便宜。"
                "好生意不等于任何价格都值得买。"
            )

    else:

        duan = (
            "先确认是不是一个真正值得长期持有的生意，"
            "不要因为便宜就自动把它当成好公司。"
        )

    result["段永平"] = duan

    # Lynch
    if (
        not np.isnan(growth)
        and not np.isnan(pe)
        and pe > 0
    ):

        growth_percent = growth * 100
        peg_like = pe / max(
            growth_percent,
            1
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
                "估值可能跑在增长之前。"
            )

    else:

        lynch = (
            "增长与估值数据不足。"
        )

    result["Lynch"] = lynch

    # Fisher
    result["Fisher"] = (
        "长期成长需要继续观察："
        "研发、产品、市场空间、竞争优势以及管理层执行力。"
    )

    return result


# ============================================================
# DEVIL'S ADVOCATE
# ============================================================

def devil_advocate(
    info,
    metrics,
    score,
    valuation
):
    arguments = []

    pe = metrics["pe"]
    growth = metrics["earnings_growth"]
    revenue = metrics["revenue_growth"]
    debt = metrics["debt_equity"]
    fcf = metrics["fcf"]

    if score >= 80:
        arguments.append(
            "市场可能已经充分认识到了公司的优秀，导致未来回报低于公司增长。"
        )

    if (
        not np.isnan(pe)
        and pe > 35
    ):
        arguments.append(
            "高估值意味着增长稍微不及预期，估值倍数就可能下降。"
        )

    if (
        not np.isnan(growth)
        and growth < 0
    ):
        arguments.append(
            "盈利负增长可能意味着市场看到了一些尚未反映在估值里的问题。"
        )

    if (
        not np.isnan(revenue)
        and revenue < 0.05
    ):
        arguments.append(
            "收入增长偏慢可能削弱长期复利能力。"
        )

    if (
        not np.isnan(debt)
        and debt > 150
    ):
        arguments.append(
            "较高杠杆会放大经济周期中的经营风险。"
        )

    if (
        np.isnan(fcf)
        or fcf <= 0
    ):
        arguments.append(
            "自由现金流不足会降低长期估值的可靠性。"
        )

    arguments.append(
        "竞争对手可能通过价格、产品或技术改变竞争格局。"
    )

    arguments.append(
        "利率变化可能改变市场愿意支付的估值倍数。"
    )

    arguments.append(
        "管理层资本配置错误可能破坏原本不错的商业模式。"
    )

    return arguments[:8]


# ============================================================
# SCENARIO
# ============================================================

def scenario_analysis(
    price,
    valuation
):
    if valuation is None:
        return None

    fair = valuation["fair"]

    return {
        "Bear": fair * 0.75,
        "Base": fair,
        "Bull": fair * 1.25
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
            ("观察仓", "10%", "估值数据不足，先不要重仓。")
        ]

    strong = valuation["strong_buy"]
    buy = valuation["buy"]
    fair = valuation["fair"]

    if price <= strong:

        if risk_preference == "保守":
            return [
                ("第一笔", "20%", "极具安全边际"),
                ("第二笔", "15%", "价格进一步确认"),
                ("第三笔", "15%", "基本面没有恶化")
            ]

        if risk_preference == "进取":
            return [
                ("第一笔", "30%", "极具安全边际"),
                ("第二笔", "25%", "继续确认"),
                ("第三笔", "20%", "基本面确认")
            ]

        return [
            ("第一笔", "25%", "强安全边际"),
            ("第二笔", "20%", "价格继续有吸引力"),
            ("第三笔", "15%", "基本面确认")
        ]

    if price <= buy:
        return [
            ("第一笔", "15%", "小仓位试探"),
            ("第二笔", "15%", "价格继续下降"),
            ("第三笔", "10%", "基本面确认")
        ]

    if price <= fair:
        return [
            ("观察仓", "5%", "合理价格"),
            ("等待", "现金", "等待安全边际")
        ]

    return [
        ("不追涨", "0%", "当前价格缺乏安全边际"),
        ("等待", "现金", "等待更好的价格")
    ]


# ============================================================
# NEWS
# ============================================================

def extract_news(news):
    rows = []

    if not isinstance(news, list):
        return rows

    for item in news[:10]:

        try:
            content = item.get(
                "content",
                item
            )

            title = content.get(
                "title",
                ""
            )

            publisher = content.get(
                "provider",
                {}
            )

            if isinstance(
                publisher,
                dict
            ):
                publisher_name = publisher.get(
                    "displayName",
                    ""
                )
            else:
                publisher_name = str(
                    publisher
                )

            if title:
                rows.append({
                    "Title": title,
                    "Publisher": publisher_name
                })

        except Exception:
            continue

    return rows


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

    <div class="hero-subtitle">
    SIMON STOCK V6.0 · ULTIMATE FREE INTELLIGENCE
    </div>

    <div class="hero-title">
    🧠 Simon Stock
    </div>

    <div class="hero-subtitle">
    Quality × Price × Safety Margin × Inversion
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

    st.caption(
        "Simon Stock V6.0"
    )

    st.caption(
        "免费智能投资研究工具"
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
        "Simon 正在读取市场数据..."
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


if history is None or history.empty:

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

price = technical.get(
    "price",
    np.nan
)

score, dimension_scores = calculate_simon_score(
    metrics,
    info,
    technical
)

valuation = fair_value_engine(
    info,
    price
)

verdict, verdict_reason = generate_verdict(
    score,
    price,
    valuation
)

masters = master_council(
    info,
    metrics,
    score,
    price,
    valuation
)

devil = devil_advocate(
    info,
    metrics,
    score,
    valuation
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
        price / previous_price - 1
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
    pe = metrics["pe"]

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
            metrics["revenue_growth"]
        )
    )

with c6:
    st.metric(
        "Market Cap",
        dollar(
            metrics["market_cap"]
        )
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
        "🧨 Devil's Advocate",
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
            SIMON SCORE
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

    st.markdown(
        "## 📊 Simon Dimensions"
    )

    dimension_cols = st.columns(6)

    dimension_list = list(
        dimension_scores.items()
    )

    for column, item in zip(
        dimension_cols,
        dimension_list
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

    st.markdown(
        "## 🏢 Company"
    )

    company_description = info.get(
        "longBusinessSummary",
        "暂无公司简介。"
    )

    st.write(
        company_description
    )

    st.caption(
        f"Sector: {sector} · Industry: {industry}"
    )


# ============================================================
# SIMON INTELLIGENCE
# ============================================================

with tabs[1]:

    st.markdown(
        "## 🧠 Simon Intelligence Engine"
    )

    st.info(
        "核心思想：不要只问“公司好不好”，还要问“这个价格是否值得”。"
    )

    quality_score = (
        dimension_scores["Business Quality"]
    )

    growth_score = (
        dimension_scores["Growth"]
    )

    profit_score = (
        dimension_scores["Profitability"]
    )

    financial_score = (
        dimension_scores["Financial Strength"]
    )

    valuation_score = (
        dimension_scores["Valuation"]
    )

    risk_score = (
        dimension_scores["Risk"]
    )

    table = pd.DataFrame(
        {
            "模块": [
                "商业质量",
                "成长",
                "盈利能力",
                "财务健康",
                "估值",
                "风险"
            ],
            "Simon评分": [
                quality_score,
                growth_score,
                profit_score,
                financial_score,
                valuation_score,
                risk_score
            ]
        }
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        "### 🎯 Simon 三大问题"
    )

    questions = [
        "这是好生意吗？",
        "这个价格值得买吗？",
        "如果我判断错了，最可能错在哪里？"
    ]

    for question in questions:

        st.markdown(
            f"☐ **{question}**"
        )

    st.divider()

    st.markdown(
        "### 🧩 Simon 投资哲学"
    )

    st.write(
        """
        **第一原则：好公司不等于好股票。**

        **第二原则：价格决定回报率。**

        **第三原则：安全边际比预测更重要。**

        **第四原则：先找自己可能错在哪里。**

        **第五原则：现金也是一种选择权。**
        """
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
            "当前公司缺少足够的现金流/盈利数据，无法可靠建立估值模型。"
        )

    else:

        fair = valuation["fair"]

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
                f"${fair:.2f}"
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
            "### 📉 Discount / Premium"
        )

        difference = (
            price / fair - 1
        )

        st.metric(
            "相对模型合理价值",
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
                "估值明显偏高。"
            )

        st.markdown(
            "### 📊 DCF Scenarios"
        )

        if "dcf" in valuation:

            dcf = valuation["dcf"]

            dcf_table = pd.DataFrame(
                {
                    "情景": [
                        "Bear",
                        "Base",
                        "Bull"
                    ],
                    "估值": [
                        dcf["Bear"],
                        dcf["Base"],
                        dcf["Bull"]
                    ]
                }
            )

            dcf_table["距离当前价格"] = (
                (
                    dcf_table["估值"] /
                    price -
                    1
                ) * 100
            ).round(1).astype(str) + "%"

            st.dataframe(
                dcf_table,
                use_container_width=True,
                hide_index=True
            )

        st.markdown(
            "### 🎯 Simon Buy Strategy"
        )

        plan_table = pd.DataFrame(
            plan,
            columns=[
                "动作",
                "资金比例",
                "逻辑"
            ]
        )

        st.dataframe(
            plan_table,
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
        "这里不是模拟这些投资人的真实观点，而是把他们公开、广为人知的投资原则转化成检查框架。"
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
        "管理层会不会把赚到的钱合理地配置？",
        "现在的价格是否给我足够安全边际？"
    ]

    for index, question in enumerate(
        master_questions,
        1
    ):

        st.markdown(
            f"**{index}.** {question}"
        )


# ============================================================
# DEVIL
# ============================================================

with tabs[4]:

    st.markdown(
        "## 🧨 Devil's Advocate"
    )

    st.error(
        "这部分故意站在“卖出 / 不买”的角度。"
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
        "### 🔥 What Would Break The Thesis?"
    )

    break_thesis = [
        "收入连续几个季度低于预期",
        "利润率持续下降",
        "自由现金流恶化",
        "核心产品竞争力下降",
        "管理层资本配置明显恶化",
        "行业出现结构性颠覆",
        "估值远远跑在基本面前面"
    ]

    for item in break_thesis:

        st.markdown(
            f"☐ {item}"
        )


# ============================================================
# TECHNICAL
# ============================================================

with tabs[5]:

    st.markdown(
        "## 📈 Technical Intelligence"
    )

    t1, t2, t3, t4 = st.columns(4)

    with t1:
        st.metric(
            "RSI",
            (
                f"{technical['rsi']:.1f}"
                if "rsi" in technical
                and not np.isnan(
                    technical["rsi"]
                )
                else "N/A"
            )
        )

    with t2:
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

    with t3:
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

    with t4:
        st.metric(
            "SMA 200",
            (
                f"${technical['sma200']:.2f}"
                if "sma200" in technical
                and not np.isnan(
                    technical["sma200"]
                )
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
        "### 📉 Drawdown"
    )

    drawdown = technical.get(
        "max_drawdown",
        np.nan
    )

    st.metric(
        "历史区间最大回撤",
        (
            f"{drawdown * 100:.1f}%"
            if not np.isnan(drawdown)
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

with tabs[6]:

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
            for item in battle_input.split(",")
            if item.strip()
        ]

        battle_symbols = list(
            dict.fromkeys(
                battle_symbols
            )
        )[:5]

        battle_rows = []

        progress = st.progress(0)

        total_symbols = len(
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

                sc, _ = calculate_simon_score(
                    met,
                    inf,
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

                upside = np.nan

                if va is not None:

                    fair_price = va["fair"]

                    upside = (
                        fair_price /
                        p -
                        1
                    )

                battle_rows.append(
                    {
                        "Ticker": battle_symbol,
                        "Price": round(p, 2),
                        "Simon Score": sc,
                        "Fair Value": (
                            round(
                                fair_price,
                                2
                            )
                            if not np.isnan(
                                fair_price
                            )
                            else np.nan
                        ),
                        "Upside": (
                            f"{upside * 100:+.1f}%"
                            if not np.isnan(
                                upside
                            )
                            else "N/A"
                        )
                    }
                )

            except Exception:
                pass

            progress.progress(
                (index + 1) /
                max(total_symbols, 1)
            )

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
                use_container_width=True,
                hide_index=True
            )

            winner = battle_df.iloc[0]

            st.success(
                f"🏆 Simon Winner："
                f"{winner['Ticker']} · "
                f"{winner['Simon Score']}/100"
            )

        else:

            st.error(
                "没有成功读取 Battle 数据。"
            )


# ============================================================
# PORTFOLIO
# ============================================================

with tabs[7]:

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

        lines = portfolio_text.splitlines()

        for line in lines:

            parts = [
                part.strip()
                for part in line.split(",")
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

                pnl_pct = (
                    pnl /
                    invested
                    if invested != 0
                    else 0
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

            if len(portfolio_df) <= 2:

                st.warning(
                    "⚠️ 持仓集中度较高。"
                )

            else:

                st.info(
                    "组合已经有一定分散度，但分散不是越多越好。"
                )

        else:

            st.error(
                "没有读取到有效持仓。"
            )


# ============================================================
# NEWS
# ============================================================

with tabs[8]:

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

            if item["Publisher"]:
                st.caption(
                    item["Publisher"]
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
    [1, 1]
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
        商业质量：{dimension_scores["Business Quality"]}/100
        </p>

        <p>
        成长：{dimension_scores["Growth"]}/100
        </p>

        <p>
        盈利：{dimension_scores["Profitability"]}/100
        </p>

        <p>
        财务：{dimension_scores["Financial Strength"]}/100
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

    st.markdown(
        f"""
        <div class="card">

        <div class="small">
        PRICE DISCIPLINE
        </div>

        <h2>{verdict}</h2>

        <p>
        当前价格：${price:.2f}
        </p>

        <p>
        Simon Fair Value：{fair_text}
        </p>

        <p>
        投资期限：{horizon}
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


st.caption(
    "Simon Stock V6.0 · Data powered by Yahoo Finance"
)

st.caption(
    "This application is for investment research and education only. "
    "It is not financial advice and does not guarantee investment returns."
)

st.caption(
    f"Last analysis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)