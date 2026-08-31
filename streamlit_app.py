import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================
# SIMON STOCK V5.1 — FREE INTELLIGENCE
# ============================================================

st.set_page_config(
    page_title="Simon Stock V5.1",
    page_icon="🧠",
    layout="wide"
)

# -------------------- STYLE --------------------

st.markdown("""
<style>
.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
}

.hero {
    padding: 28px;
    border-radius: 24px;
    margin-bottom: 22px;
    border: 1px solid rgba(128,128,128,.22);
    background: linear-gradient(
        135deg,
        rgba(80,100,180,.12),
        rgba(150,80,180,.08)
    );
}

.hero-title {
    font-size: 42px;
    font-weight: 900;
}

.hero-subtitle {
    opacity: .65;
    font-size: 16px;
}

.section-title {
    font-size: 25px;
    font-weight: 850;
    margin-top: 20px;
    margin-bottom: 12px;
}

.big-score {
    font-size: 58px;
    font-weight: 900;
}

.verdict {
    font-size: 28px;
    font-weight: 850;
}

.card {
    padding: 20px;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,.18);
    background: rgba(128,128,128,.05);
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================

def num(x, default=np.nan):
    try:
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return default
        return x
    except Exception:
        return default


def pct(x):
    x = num(x)
    if np.isnan(x):
        return "N/A"
    return f"{x * 100:.1f}%"


def money(x):
    x = num(x)

    if np.isnan(x):
        return "N/A"

    if abs(x) >= 1e12:
        return f"${x/1e12:.2f}T"

    if abs(x) >= 1e9:
        return f"${x/1e9:.2f}B"

    if abs(x) >= 1e6:
        return f"${x/1e6:.2f}M"

    return f"${x:,.2f}"


# ============================================================
# DATA
# ============================================================

@st.cache_data(ttl=300)
def get_data(symbol, period):

    stock = yf.Ticker(symbol)

    history = stock.history(
        period=period,
        interval="1d",
        auto_adjust=False
    )

    try:
        info = stock.info
    except Exception:
        info = {}

    return history, info


# ============================================================
# TECHNICALS
# ============================================================

def technicals(history):

    result = {}

    if history.empty:
        return result

    close = history["Close"].dropna()

    if len(close) < 20:
        return result

    result["price"] = close.iloc[-1]

    result["sma20"] = close.rolling(20).mean().iloc[-1]

    if len(close) >= 50:
        result["sma50"] = close.rolling(50).mean().iloc[-1]
    else:
        result["sma50"] = np.nan

    if len(close) >= 200:
        result["sma200"] = close.rolling(200).mean().iloc[-1]
    else:
        result["sma200"] = np.nan

    if len(close) >= 15:

        delta = close.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)

        result["rsi"] = (
            100 - 100 / (1 + rs)
        ).iloc[-1]

    if len(close) >= 30:

        result["return30"] = (
            close.iloc[-1] /
            close.iloc[-30]
            - 1
        )

    if len(close) >= 252:

        result["return1y"] = (
            close.iloc[-1] /
            close.iloc[-252]
            - 1
        )

    return result


# ============================================================
# SIMON SCORE
# ============================================================

def simon_score(info):

    categories = {
        "Business Quality": 50,
        "Growth": 50,
        "Profitability": 50,
        "Financial Strength": 50,
        "Valuation": 50,
        "Risk": 50
    }

    # Growth
    revenue_growth = num(
        info.get("revenueGrowth")
    )

    earnings_growth = num(
        info.get("earningsGrowth")
    )

    if not np.isnan(revenue_growth):

        if revenue_growth >= .25:
            categories["Growth"] += 30

        elif revenue_growth >= .15:
            categories["Growth"] += 20

        elif revenue_growth >= .05:
            categories["Growth"] += 8

        elif revenue_growth < 0:
            categories["Growth"] -= 25

    if not np.isnan(earnings_growth):

        if earnings_growth >= .25:
            categories["Growth"] += 25

        elif earnings_growth >= .15:
            categories["Growth"] += 15

        elif earnings_growth < 0:
            categories["Growth"] -= 20

    # Profitability
    roe = num(
        info.get("returnOnEquity")
    )

    margin = num(
        info.get("profitMargins")
    )

    if not np.isnan(roe):

        if roe >= .30:
            categories["Profitability"] += 35

        elif roe >= .20:
            categories["Profitability"] += 25

        elif roe >= .10:
            categories["Profitability"] += 10

        elif roe < 0:
            categories["Profitability"] -= 25

    if not np.isnan(margin):

        if margin >= .25:
            categories["Profitability"] += 25

        elif margin >= .15:
            categories["Profitability"] += 15

        elif margin < 0:
            categories["Profitability"] -= 25

    # Financial strength
    fcf = num(
        info.get("freeCashflow")
    )

    debt = num(
        info.get("debtToEquity")
    )

    if not np.isnan(fcf):

        if fcf > 0:
            categories["Financial Strength"] += 20
        else:
            categories["Financial Strength"] -= 20

    if not np.isnan(debt):

        if debt < 50:
            categories["Financial Strength"] += 20

        elif debt < 100:
            categories["Financial Strength"] += 8

        elif debt > 200:
            categories["Financial Strength"] -= 25

    # Valuation
    pe = num(
        info.get("trailingPE")
    )

    if not np.isnan(pe) and pe > 0:

        if pe < 18:
            categories["Valuation"] += 30

        elif pe < 25:
            categories["Valuation"] += 18

        elif pe < 35:
            categories["Valuation"] += 5

        elif pe < 50:
            categories["Valuation"] -= 15

        else:
            categories["Valuation"] -= 30

    # Risk
    beta = num(
        info.get("beta")
    )

    if not np.isnan(beta):

        if beta < 1:
            categories["Risk"] += 15

        elif beta > 2:
            categories["Risk"] -= 25

    # Business size
    market_cap = num(
        info.get("marketCap")
    )

    if not np.isnan(market_cap):

        if market_cap > 100e9:
            categories["Business Quality"] += 25

        elif market_cap > 10e9:
            categories["Business Quality"] += 15

        elif market_cap > 1e9:
            categories["Business Quality"] += 5

    for k in categories:

        categories[k] = max(
            0,
            min(
                100,
                categories[k]
            )
        )

    weights = {
        "Business Quality": .20,
        "Growth": .18,
        "Profitability": .18,
        "Financial Strength": .16,
        "Valuation": .18,
        "Risk": .10
    }

    total = sum(
        categories[k] * weights[k]
        for k in categories
    )

    return int(round(total)), categories


# ============================================================
# VALUATION
# ============================================================

def valuation(info, price):

    pe = num(
        info.get("trailingPE")
    )

    forward_pe = num(
        info.get("forwardPE")
    )

    earnings_growth = num(
        info.get("earningsGrowth")
    )

    estimates = []

    # Current PE normalized
    if not np.isnan(pe) and pe > 0:

        if pe < 20:
            target_multiple = 22

        elif pe < 30:
            target_multiple = 25

        elif pe < 45:
            target_multiple = 28

        else:
            target_multiple = 30

        estimates.append(
            price *
            target_multiple /
            pe
        )

    # Forward PE
    if not np.isnan(forward_pe) and forward_pe > 0:

        estimates.append(
            price *
            24 /
            forward_pe
        )

    if estimates:

        fair = np.median(
            estimates
        )

    else:

        fair = price

    # Growth adjustment
    if not np.isnan(earnings_growth):

        if earnings_growth > .25:
            fair *= 1.08

        elif earnings_growth > .15:
            fair *= 1.04

        elif earnings_growth < 0:
            fair *= .90

    fair = max(
        price * .50,
        min(
            price * 1.80,
            fair
        )
    )

    return {
        "strong_buy": fair * .70,
        "buy": fair * .82,
        "fair": fair,
        "expensive": fair * 1.20,
        "bubble": fair * 1.45
    }


# ============================================================
# VERDICT
# ============================================================

def verdict(score, price, val):

    fair = val["fair"]

    if price <= val["strong_buy"]:

        if score >= 75:
            return "🟢 STRONG BUY"

        return "🟡 VALUE OPPORTUNITY"

    if price <= val["buy"]:

        if score >= 75:
            return "🟢 BUY"

        return "🟡 WATCH"

    if price <= fair:

        if score >= 85:
            return "🟡 GREAT COMPANY / FAIR PRICE"

        return "🟡 WATCH"

    if price <= val["expensive"]:

        return "🟠 EXPENSIVE"

    return "🔴 AVOID / WAIT"


# ============================================================
# MASTER THINKING
# ============================================================

def master_framework(
    score,
    info,
    price,
    val
):

    growth = num(
        info.get("earningsGrowth")
    )

    roe = num(
        info.get("returnOnEquity")
    )

    pe = num(
        info.get("trailingPE")
    )

    fcf = num(
        info.get("freeCashflow")
    )

    # Buffett
    if score >= 80:

        buffett = (
            "商业质量较强。重点问题不是公司够不够好，"
            "而是当前价格是否给长期回报留下足够安全边际。"
        )

    else:

        buffett = (
            "目前证据不足以证明这是一个可以长期安心持有的生意。"
        )

    # Munger
    risks = []

    if not np.isnan(pe) and pe > 40:
        risks.append("估值压缩")

    if not np.isnan(growth) and growth < 0:
        risks.append("增长恶化")

    if not np.isnan(roe) and roe < .10:
        risks.append("资本效率偏低")

    if np.isnan(fcf) or fcf <= 0:
        risks.append("现金流质量")

    if not risks:
        risks.append("竞争格局变化")

    munger = (
        "反过来思考：最大的风险可能来自 "
        + "、".join(risks)
        + "。"
    )

    # Duan Yongping
    if score >= 80 and price <= val["fair"]:

        duan = (
            "Right Business / Right People / Right Price "
            "三个条件目前相对协调。"
        )

    elif score >= 80:

        duan = (
            "Right Business 可能不错，但 Right Price "
            "目前是主要问题。"
        )

    else:

        duan = (
            "目前仍需要更多证据证明这是值得长期持有的好生意。"
        )

    # Lynch
    if (
        not np.isnan(growth)
        and not np.isnan(pe)
        and pe > 0
    ):

        peg_like = pe / max(
            growth * 100,
            1
        )

        if peg_like < 1:
            lynch = "增长相对估值具有吸引力。"

        elif peg_like < 2:
            lynch = "增长和估值基本匹配。"

        else:
            lynch = "估值可能跑在增长前面。"

    else:

        lynch = "数据不足以进行有效的增长/估值比较。"

    # Fisher
    fisher = (
        "长期成长需要继续观察研发、产品、市场空间和管理层执行力。"
    )

    return {
        "Buffett": buffett,
        "Munger": munger,
        "段永平": duan,
        "Lynch": lynch,
        "Fisher": fisher
    }


# ============================================================
# DEVIL'S ADVOCATE
# ============================================================

def devil's_advocate(
    info,
    price,
    score,
    val
):

    arguments = []

    pe = num(
        info.get("trailingPE")
    )

    growth = num(
        info.get("earningsGrowth")
    )

    revenue = num(
        info.get("revenueGrowth")
    )

    debt = num(
        info.get("debtToEquity")
    )

    fcf = num(
        info.get("freeCashflow")
    )

    if score >= 80:
        arguments.append(
            "高质量公司可能已经被市场充分定价。"
        )

    if not np.isnan(pe) and pe > 35:
        arguments.append(
            "高估值意味着未来增长一旦低于预期，估值可能快速下降。"
        )

    if not np.isnan(growth) and growth < .10:
        arguments.append(
            "盈利增长不足可能无法支撑高估值。"
        )

    if not np.isnan(revenue) and revenue < .10:
        arguments.append(
            "收入增长放缓可能意味着长期增长空间重新定价。"
        )

    if not np.isnan(debt) and debt > 150:
        arguments.append(
            "较高杠杆可能放大经营压力。"
        )

    if np.isnan(fcf) or fcf <= 0:
        arguments.append(
            "自由现金流不足会降低估值模型可靠性。"
        )

    arguments.append(
        "行业竞争格局可能发生变化。"
    )

    arguments.append(
        "宏观经济和利率变化可能影响市场给予的估值倍数。"
    )

    return arguments[:7]


# ============================================================
# BUY PLAN
# ============================================================

def buy_plan(price, val, score):

    if price <= val["strong_buy"]:

        return (
            "🟢 强安全边际",
            "如果基本面没有恶化，可以考虑分批建立仓位。"
        )

    if price <= val["buy"]:

        return (
            "🟢 有吸引力",
            "可以考虑小仓位开始，并保留现金应对波动。"
        )

    if price <= val["fair"]:

        return (
            "🟡 合理",
            "公司可以不错，但没有必要因为害怕踏空而重仓。"
        )

    if price <= val["expensive"]:

        return (
            "🟠 偏贵",
            "等待更好的价格通常比追涨更有优势。"
        )

    return (
        "🔴 很贵",
        "除非未来增长明显超预期，否则安全边际不足。"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<div class="hero-subtitle">
SIMON STOCK V5.1 · FREE INTELLIGENCE
</div>

<div class="hero-title">
🧠 Simon Stock
</div>

<div class="hero-subtitle">
Think Like an Owner · Price Matters · Invert Before You Invest
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🔎 Stock Research")

    symbol = st.text_input(
        "股票代码",
        "AAPL"
    ).upper().strip()

    period = st.selectbox(
        "数据周期",
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

    st.header("🎯 投资偏好")

    risk_preference = st.selectbox(
        "风险偏好",
        [
            "保守",
            "平衡",
            "进取"
        ]
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


# ============================================================
# LOAD
# ============================================================

if not symbol:

    st.warning(
        "请输入股票代码。"
    )

    st.stop()

try:

    with st.spinner(
        f"正在分析 {symbol}..."
    ):

        history, info = get_data(
            symbol,
            period
        )

except Exception as e:

    st.error(
        "读取股票数据失败。"
    )

    st.code(
        str(e)
    )

    st.stop()


if history.empty:

    st.error(
        "没有找到有效股票数据。"
    )

    st.stop()


price = num(
    history["Close"].iloc[-1]
)

previous = (
    num(history["Close"].iloc[-2])
    if len(history) >= 2
    else np.nan
)

daily_change = (
    price / previous - 1
    if not np.isnan(previous)
    and previous != 0
    else np.nan
)

company = info.get(
    "longName",
    symbol
)

score, dimensions = simon_score(
    info
)

val = valuation(
    info,
    price
)

final_verdict = verdict(
    score,
    price,
    val
)

tech = technicals(
    history
)

masters = master_framework(
    score,
    info,
    price,
    val
)

devil = devil's_advocate(
    info,
    price,
    score,
    val
)

buy_label, buy_text = buy_plan(
    price,
    val,
    score
)


# ============================================================
# TOP
# ============================================================

st.subheader(
    f"{company} · {symbol}"
)

c1, c2, c3, c4, c5 = st.columns(5)

with c1:

    st.metric(
        "Price",
        f"${price:.2f}",
        (
            f"{daily_change*100:+.2f}%"
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

    pe = num(
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

    roe = num(
        info.get("returnOnEquity")
    )

    st.metric(
        "ROE",
        pct(roe)
    )

with c5:

    market_cap = num(
        info.get("marketCap")
    )

    st.metric(
        "Market Cap",
        money(market_cap)
    )


# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "🏠 Dashboard",
    "🧠 Simon Intelligence",
    "🏆 Master Council",
    "🎯 Buy Zone",
    "🧨 Devil's Advocate",
    "⚔️ Battle",
    "💼 Portfolio"
])


# ============================================================
# DASHBOARD
# ============================================================

with tabs[0]:

    st.markdown(
        '<div class="section-title">🧠 Simon Verdict</div>',
        unsafe_allow_html=True
    )

    a, b = st.columns(
        [1, 2]
    )

    with a:

        st.markdown(
            f"""
            <div class="card">

            <div>
            SIMON SCORE
            </div>

            <div class="big-score">
            {score}
            </div>

            <div>
            /100
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with b:

        st.markdown(
            f"""
            <div class="card">

            <div>
            CURRENT DECISION
            </div>

            <div class="verdict">
            {final_verdict}
            </div>

            <p>
            当前价格：
            <b>${price:.2f}</b>
            </p>

            <p>
            模型合理价值：
            <b>${val['fair']:.2f}</b>
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">📈 Price Chart</div>',
        unsafe_allow_html=True
    )

    chart = history[
        ["Close"]
    ].rename(
        columns={
            "Close": symbol
        }
    )

    st.line_chart(
        chart,
        height=420
    )

    st.markdown(
        '<div class="section-title">📊 Simon Dimensions</div>',
        unsafe_allow_html=True
    )

    score_table = pd.DataFrame({
        "Dimension":
            list(dimensions.keys()),
        "Score":
            list(dimensions.values())
    })

    st.dataframe(
        score_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# INTELLIGENCE
# ============================================================

with tabs[1]:

    st.subheader(
        "🧠 Simon Intelligence"
    )

    st.markdown(
        """
        ### 核心原则

        **公司质量 × 当前价格 × 安全边际**

        Simon 不追求：

        > “预测明天涨还是跌。”

        Simon 更关心：

        > **“如果我持有 3–5 年，现在这个价格是否值得承担风险？”**
        """
    )

    st.divider()

    quality = dimensions[
        "Business Quality"
    ]

    growth = dimensions[
        "Growth"
    ]

    profitability = dimensions[
        "Profitability"
    ]

    financial = dimensions[
        "Financial Strength"
    ]

    valuation_score = dimensions[
        "Valuation"
    ]

    risk_score = dimensions[
        "Risk"
    ]

    cols = st.columns(6)

    values = [
        ("Business", quality),
        ("Growth", growth),
        ("Profit", profitability),
        ("Financial", financial),
        ("Valuation", valuation_score),
        ("Risk", risk_score)
    ]

    for col, (name, value) in zip(
        cols,
        values
    ):

        col.metric(
            name,
            f"{value}/100"
        )

    st.divider()

    st.subheader(
        "🎯 Price × Quality Matrix"
    )

    if score >= 80 and price <= val["buy"]:

        st.success(
            "🔥 高质量 + 有安全边际：这是 Simon 最喜欢的组合。"
        )

    elif score >= 80 and price > val["fair"]:

        st.warning(
            "💎 高质量公司，但价格已经明显影响回报率。"
        )

    elif score < 65 and price <= val["buy"]:

        st.warning(
            "🟡 便宜不代表优秀：需要确认是不是价值陷阱。"
        )

    else:

        st.info(
            "🟡 当前处于中间区域，耐心等待更多安全边际。"
        )


# ============================================================
# MASTER COUNCIL
# ============================================================

with tabs[2]:

    st.subheader(
        "🏆 Investment Master Council"
    )

    for name, text in masters.items():

        with st.expander(
            f"{name} Lens",
            expanded=True
        ):

            st.write(text)

    st.divider()

    st.subheader(
        "🎯 Five Master Questions"
    )

    questions = [
        "如果市场关闭五年，我还愿意持有吗？",
        "如果我不能卖出股票，我还愿意买这个生意吗？",
        "什么事情最可能让我永久亏钱？",
        "这是好生意、好管理层、好价格吗？",
        "未来五年公司本身会不会比现在更强？"
    ]

    for q in questions:

        st.markdown(
            f"☐ **{q}**"
        )


# ============================================================
# BUY ZONE
# ============================================================

with tabs[3]:

    st.subheader(
        "🎯 Simon Buy Zone"
    )

    b1, b2, b3, b4, b5 = st.columns(5)

    b1.metric(
        "🟢 Strong Buy",
        f"${val['strong_buy']:.2f}"
    )

    b2.metric(
        "🟢 Buy",
        f"${val['buy']:.2f}"
    )

    b3.metric(
        "🟡 Fair",
        f"${val['fair']:.2f}"
    )

    b4.metric(
        "🟠 Expensive",
        f"${val['expensive']:.2f}"
    )

    b5.metric(
        "🔴 Bubble",
        f"${val['bubble']:.2f}"
    )

    st.divider()

    st.markdown(
        f"""
        ### {buy_label}

        {buy_text}
        """
    )

    st.info(
        "价格区间是模型估计，不是精确买卖点。"
    )

    st.subheader(
        "💡 Simon 资金纪律"
    )

    st.write(
        """
        不要因为“怕错过”一次性满仓。

        更合理的思路：

        **第一笔 → 试仓**

        **第二笔 → 价格进一步有吸引力**

        **第三笔 → 基本面确认 + 更大安全边际**

        **现金 → 永远保留选择权**
        """
    )


# ============================================================
# DEVIL
# ============================================================

with tabs[4]:

    st.subheader(
        "🧨 Devil's Advocate"
    )

    st.warning(
        "这一页故意和你的投资观点唱反调。"
    )

    for i, argument in enumerate(
        devil,
        1
    ):

        st.markdown(
            f"### {i}. {argument}"
        )

    st.divider()

    st.subheader(
        "🧠 Inversion"
    )

    st.write(
        """
        如果 Simon 判断错了：

        **最可能不是因为股价突然跌了一天。**

        而是因为：

        1. 公司护城河被削弱
        2. 长期增长低于预期
        3. 管理层资本配置出现问题
        4. 估值过高导致回报被压缩
        5. 行业发生结构性变化
        """
    )


# ============================================================
# BATTLE
# ============================================================

with tabs[5]:

    st.subheader(
        "⚔️ Simon Stock Battle"
    )

    battle_input = st.text_input(
        "输入 2–4 个股票",
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
            dict.fromkeys(symbols)
        )[:4]

        results = []

        with st.spinner(
            "Simon 正在进行股票 Battle..."
        ):

            for s in symbols:

                try:

                    h, inf = get_data(
                        s,
                        "1y"
                    )

                    if h.empty:
                        continue

                    p = num(
                        h["Close"].iloc[-1]
                    )

                    sc, _ = simon_score(
                        inf
                    )

                    va = valuation(
                        inf,
                        p
                    )

                    upside = (
                        va["fair"] /
                        p -
                        1
                    )

                    results.append({

                        "Ticker": s,

                        "Price":
                            round(p, 2),

                        "Simon Score":
                            sc,

                        "Fair Value":
                            round(
                                va["fair"],
                                2
                            ),

                        "Upside":
                            f"{upside*100:+.1f}%"

                    })

                except Exception:
                    continue

        if results:

            battle = pd.DataFrame(
                results
            ).sort_values(
                "Simon Score",
                ascending=False
            )

            st.dataframe(
                battle,
                use_container_width=True,
                hide_index=True
            )

            winner = battle.iloc[0]

            st.success(
                f"🏆 Simon Winner："
                f"{winner['Ticker']} · "
                f"{winner['Simon Score']}/100"
            )

        else:

            st.error(
                "Battle 数据读取失败。"
            )


# ============================================================
# PORTFOLIO
# ============================================================

with tabs[6]:

    st.subheader(
        "💼 Simon Portfolio Brain"
    )

    st.write(
        """
        格式：

        `股票代码,股数,成本价`
        """
    )

    portfolio_input = st.text_area(
        "Portfolio",
        """AAPL,2,310
GOOGL,2,342
AVGO,2,352"""
    )

    if st.button(
        "🧠 Analyze Portfolio",
        type="primary"
    ):

        rows = []

        for line in portfolio_input.splitlines():

            parts = [
                x.strip()
                for x in line.split(",")
            ]

            if len(parts) < 3:
                continue

            try:

                s = parts[0].upper()

                shares = float(
                    parts[1]
                )

                cost = float(
                    parts[2]
                )

                h, inf = get_data(
                    s,
                    "5d"
                )

                if h.empty:
                    continue

                p = num(
                    h["Close"].iloc[-1]
                )

                value = (
                    p *
                    shares
                )

                invested = (
                    cost *
                    shares
                )

                pnl = (
                    value -
                    invested
                )

                pnl_pct = (
                    pnl /
                    invested *
                    100
                    if invested
                    else 0
                )

                rows.append({

                    "Ticker": s,

                    "Shares":
                        shares,

                    "Cost":
                        cost,

                    "Price":
                        round(
                            p,
                            2
                        ),

                    "Value":
                        round(
                            value,
                            2
                        ),

                    "P/L":
                        round(
                            pnl,
                            2
                        ),

                    "P/L %":
                        round(
                            pnl_pct,
                            2
                        )

                })

            except Exception:
                continue

        if rows:

            portfolio = pd.DataFrame(
                rows
            )

            st.dataframe(
                portfolio,
                use_container_width=True,
                hide_index=True
            )

            total_value = portfolio[
                "Value"
            ].sum()

            total_pnl = portfolio[
                "P/L"
            ].sum()

            pc1, pc2 = st.columns(2)

            pc1.metric(
                "Portfolio Value",
                f"${total_value:,.2f}"
            )

            pc2.metric(
                "Total P/L",
                f"${total_pnl:+,.2f}"
            )

            if len(portfolio) <= 2:

                st.warning(
                    "⚠️ 组合集中度较高。"
                )

            else:

                st.success(
                    "🟢 至少存在一定分散。"
                )

        else:

            st.error(
                "没有读取到有效持仓。"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    f"Simon Stock V5.1 · Updated {datetime.now().strftime('%Y-%m-%d %H:%M')}"
)

st.caption(
    "Investment research and education tool. "
    "Not financial advice. No return is guaranteed."
)