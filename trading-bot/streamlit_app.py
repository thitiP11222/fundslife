from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data import download_ohlcv
from src.risk import position_size_from_risk
from src.technical_dashboard import (
    TechnicalConfig,
    add_dashboard_indicators,
    build_trade_plan,
)
from src.unified_momentum_scanner import (
    UnifiedMomentumConfig,
    scan_unified_momentum,
)


UNIVERSE = [
    "NVDA",
    "AMD",
    "AAPL",
    "MSFT",
    "META",
    "GOOGL",
    "AMZN",
    "AVGO",
    "TSLA",
    "NFLX",
]
BENCHMARK = "QQQ"

st.set_page_config(
    page_title="Fundslife Trading Workstation",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.18);
        border-radius: 12px;
        padding: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60)
def load_data(
    ticker: str,
    interval: str,
    period_value: str,
) -> pd.DataFrame:
    return download_ohlcv(ticker, interval, period_value)


@st.cache_data(ttl=300)
def load_momentum_scan() -> tuple[pd.DataFrame, dict]:
    benchmark = download_ohlcv(
        BENCHMARK,
        "1d",
        "5y",
    )

    stocks = {
        symbol: download_ohlcv(symbol, "1d", "5y")
        for symbol in UNIVERSE
    }

    return scan_unified_momentum(
        stocks,
        benchmark,
        UnifiedMomentumConfig(),
    )


def action_label(action: str) -> str:
    labels = {
        "BUY_CANDIDATE": "Core candidate",
        "WATCH": "Watch",
        "NO_TRADE": "No trade",
    }
    return labels.get(action, action)


def setup_label(setup: str) -> str:
    labels = {
        "BUY_SETUP": "Technical setup",
        "WATCH": "Wait / monitor",
        "AVOID": "Avoid for now",
    }
    return labels.get(setup, setup)


def build_chart(
    symbol: str,
    timeframe: str,
    data: pd.DataFrame,
    plan: dict,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name=symbol,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["EMA20"],
            mode="lines",
            name="EMA20",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["EMA50"],
            mode="lines",
            name="EMA50",
        )
    )

    for level in plan["supports"]:
        fig.add_hline(
            y=level,
            line_dash="dot",
            annotation_text=f"S {level:.2f}",
            annotation_position="bottom left",
        )

    for level in plan["resistances"]:
        fig.add_hline(
            y=level,
            line_dash="dash",
            annotation_text=f"R {level:.2f}",
            annotation_position="top left",
        )

    fig.add_hline(
        y=plan["pullback_entry"],
        line_dash="dot",
        annotation_text=f"Pullback {plan['pullback_entry']:.2f}",
    )
    fig.add_hline(
        y=plan["breakout_entry"],
        line_dash="dash",
        annotation_text=f"Breakout {plan['breakout_entry']:.2f}",
    )
    fig.add_hline(
        y=plan["stop"],
        line_dash="dashdot",
        annotation_text=f"Stop {plan['stop']:.2f}",
    )
    fig.add_hline(
        y=plan["target_1"],
        line_dash="dot",
        annotation_text=f"T1 {plan['target_1']:.2f}",
    )
    fig.add_hline(
        y=plan["target_2"],
        line_dash="dot",
        annotation_text=f"T2 {plan['target_2']:.2f}",
    )

    fig.update_layout(
        height=650,
        xaxis_rangeslider_visible=False,
        title=f"{symbol} · {timeframe}",
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig


st.title("Fundslife Trading Workstation")
st.caption(
    "คัดหุ้นด้วย Momentum ก่อน แล้วค่อยใช้ Technical Analysis หาแผนเข้า–ออกและขนาด Position"
)

try:
    scanner_df, regime = load_momentum_scan()
except Exception as exc:
    st.error(f"โหลด Momentum Scanner ไม่สำเร็จ: {exc}")
    st.stop()

top1, top2, top3, market = st.columns(4)

market_state = "RISK-ON" if regime["risk_on"] else "RISK-OFF"

with market:
    st.metric(
        "Market",
        market_state,
        help=f"QQQ {regime['benchmark_close']:.2f} vs MA200 {regime['benchmark_ma200']:.2f}",
    )

candidate_rows = scanner_df.head(3)

for column, (_, row) in zip([top1, top2, top3], candidate_rows.iterrows()):
    with column:
        st.metric(
            f"#{int(row['momentum_rank'])} {row['symbol']}",
            action_label(row["fractional_action"]),
            delta=f"3M {row['momentum_3m_pct']:.1f}%",
            help=(
                f"6M momentum {row['momentum_6m_pct']:.1f}% · "
                f"ATR {row['atr_pct']:.1f}%"
            ),
        )

st.divider()

with st.sidebar:
    st.header("Trade Settings")

    account_equity = st.number_input(
        "เงินทุนในบัญชี (USD)",
        min_value=1.0,
        value=30.0,
        step=5.0,
        help="ใส่มูลค่าพอร์ตที่ต้องการใช้คำนวณ position size",
    )

    risk_pct = st.slider(
        "เสี่ยงต่อไม้ (%)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.5,
    )

    max_position_pct = st.slider(
        "เงินสูงสุดต่อ Position (%)",
        min_value=10,
        max_value=100,
        value=60,
        step=5,
    )

    st.divider()

    timeframe = st.selectbox(
        "Timeframe สำหรับหาจุดเข้า",
        ["1h", "30m", "15m", "1d"],
        index=0,
    )

    period_map = {
        "15m": "1mo",
        "30m": "1mo",
        "1h": "3mo",
        "1d": "1y",
    }
    period = period_map[timeframe]

    if st.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption(
        "Intraday ใช้แท่งที่ปิดแล้วเป็นหลัก เพราะ volume ของแท่งที่ยังไม่ปิดยังไม่สมบูรณ์"
    )


candidate_symbols = scanner_df[
    scanner_df["fractional_action"].isin(["BUY_CANDIDATE", "WATCH"])
]["symbol"].tolist()

default_options = candidate_symbols + [
    symbol for symbol in UNIVERSE
    if symbol not in candidate_symbols
]

control_left, control_right = st.columns([2, 1])

with control_left:
    selected_symbol = st.selectbox(
        "เลือกหุ้นเพื่อวิเคราะห์",
        default_options,
        index=0,
        help="เรียงหุ้น Momentum ที่น่าสนใจก่อน",
    )

with control_right:
    custom_symbol = st.text_input(
        "หรือพิมพ์ Ticker",
        placeholder="เช่น PLTR",
    ).upper().strip()

symbol = custom_symbol or selected_symbol

scanner_match = scanner_df[scanner_df["symbol"] == symbol]

if not scanner_match.empty:
    scan_row = scanner_match.iloc[0]
    badge1, badge2, badge3, badge4 = st.columns(4)

    with badge1:
        st.metric(
            "Momentum rank",
            f"#{int(scan_row['momentum_rank'])}",
        )
    with badge2:
        st.metric(
            "3M momentum",
            f"{scan_row['momentum_3m_pct']:.1f}%",
        )
    with badge3:
        st.metric(
            "6M momentum",
            f"{scan_row['momentum_6m_pct']:.1f}%",
        )
    with badge4:
        st.metric(
            "Scanner action",
            action_label(scan_row["fractional_action"]),
        )

try:
    raw = load_data(
        symbol,
        timeframe,
        period,
    )
except Exception as exc:
    st.error(f"โหลดข้อมูล {symbol} ไม่สำเร็จ: {exc}")
    st.stop()

cfg = TechnicalConfig()

try:
    data = add_dashboard_indicators(raw, cfg)
    plan = build_trade_plan(raw, cfg)
except Exception as exc:
    st.error(f"ข้อมูลไม่พอสำหรับวิเคราะห์ {symbol}: {exc}")
    st.stop()


tab_overview, tab_chart, tab_details = st.tabs(
    [
        "Trade Plan",
        "Chart",
        "Technical Details",
    ]
)

with tab_overview:
    st.subheader(f"{symbol} Trade Plan")

    a, b, c, d = st.columns(4)

    with a:
        st.metric(
            "Price",
            f"USD {plan['close']:.2f}",
        )

    with b:
        st.metric(
            "Technical",
            setup_label(plan["setup"]),
        )

    with c:
        st.metric(
            "Score",
            f"{plan['score']}/100",
        )

    with d:
        st.metric(
            "RSI",
            f"{plan['rsi']:.1f}",
        )

    st.markdown("#### เลือกแผนเข้า")

    entry_mode = st.radio(
        "Entry style",
        ["Pullback", "Breakout"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if entry_mode == "Pullback":
        entry_price = plan["pullback_entry"]
        entry_note = "รอราคาย่อกลับมาใกล้แนวรับ/EMA"
    else:
        entry_price = plan["breakout_entry"]
        entry_note = "รอราคายืนยันเหนือแนวต้านก่อน"

    shares = position_size_from_risk(
        equity=float(account_equity),
        risk_fraction=float(risk_pct) / 100.0,
        entry_price=entry_price,
        stop_price=plan["stop"],
        max_position_value_pct=float(max_position_pct) / 100.0,
        allow_fractional=True,
        share_step=0.001,
    )

    position_value = shares * entry_price
    max_loss = shares * abs(entry_price - plan["stop"])

    p1, p2, p3, p4, p5 = st.columns(5)

    with p1:
        st.metric("Entry", f"USD {entry_price:.2f}")
    with p2:
        st.metric("Stop", f"USD {plan['stop']:.2f}")
    with p3:
        st.metric("Target 1", f"USD {plan['target_1']:.2f}")
    with p4:
        st.metric("Target 2", f"USD {plan['target_2']:.2f}")
    with p5:
        st.metric("Size", f"{shares:.3f} shares")

    st.info(
        f"{entry_note} · ใช้เงินประมาณ USD {position_value:.2f} · "
        f"ขาดทุนตาม Stop ประมาณ USD {max_loss:.2f}"
    )

    st.markdown("#### Decision checklist")

    checks = []

    if not scanner_match.empty:
        checks.append(
            (
                "Momentum",
                scan_row["fractional_action"] == "BUY_CANDIDATE",
                f"Rank #{int(scan_row['momentum_rank'])}",
            )
        )

    checks.extend(
        [
            (
                "Trend",
                plan["close"] > plan["ema20"] > plan["ema50"],
                "Close > EMA20 > EMA50",
            ),
            (
                "RSI",
                50 <= plan["rsi"] <= 68,
                f"RSI {plan['rsi']:.1f}",
            ),
            (
                "Volume",
                plan["volume_ratio"] >= 1.0,
                f"{plan['volume_ratio']:.2f}x average",
            ),
            (
                "Risk",
                shares > 0,
                f"Risk {risk_pct:.1f}% / trade",
            ),
        ]
    )

    for name, passed, detail in checks:
        icon = "PASS" if passed else "WAIT"
        st.write(f"**{icon} · {name}** — {detail}")

    if plan["setup"] == "BUY_SETUP":
        st.success(
            "Technical conditions are relatively strong. "
            "Still wait for the chosen entry condition before placing a trade."
        )
    elif plan["setup"] == "WATCH":
        st.warning(
            "Setup is incomplete. Keep it on the watchlist rather than forcing an entry."
        )
    else:
        st.error(
            "Technical conditions are weak. Avoid opening a new long position for now."
        )


with tab_chart:
    st.plotly_chart(
        build_chart(
            symbol,
            timeframe,
            data,
            plan,
        ),
        use_container_width=True,
    )

    support_text = (
        ", ".join(f"USD {x:.2f}" for x in plan["supports"])
        if plan["supports"]
        else "ไม่มี"
    )

    resistance_text = (
        ", ".join(f"USD {x:.2f}" for x in plan["resistances"])
        if plan["resistances"]
        else "ไม่มี"
    )

    level_left, level_right = st.columns(2)
    with level_left:
        st.write(f"**Support:** {support_text}")
    with level_right:
        st.write(f"**Resistance:** {resistance_text}")


with tab_details:
    left, right = st.columns(2)

    with left:
        st.markdown("#### Technical indicators")

        indicator_df = pd.DataFrame(
            {
                "Indicator": [
                    "EMA20",
                    "EMA50",
                    "RSI14",
                    "ATR14",
                    "Volume ratio",
                ],
                "Value": [
                    round(plan["ema20"], 2),
                    round(plan["ema50"], 2),
                    round(plan["rsi"], 2),
                    round(plan["atr"], 2),
                    round(plan["volume_ratio"], 2),
                ],
            }
        )

        st.dataframe(
            indicator_df,
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.markdown("#### Detected patterns")

        if plan["patterns"]:
            st.dataframe(
                pd.DataFrame(plan["patterns"]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write("No supported pattern on the latest bars.")

    st.markdown("#### Why this score?")

    if plan["reasons"]:
        for reason in plan["reasons"]:
            st.write(f"- {reason}")
    else:
        st.write("- No bullish confirmation currently qualifies.")


st.divider()
st.caption(
    "Momentum Scanner = ใช้เลือกหุ้น · Technical Dashboard = ใช้วางแผนจุดเข้า/ออก · "
    "Position Size = คุมความเสี่ยง. ข้อมูล Yahoo Finance อาจมี delay และระบบไม่ส่งคำสั่งซื้อขายอัตโนมัติ."
)
