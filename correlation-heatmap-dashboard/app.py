from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="Portfolio Correlation Heatmap",
    page_icon="🧩",
    layout="wide",
)

st.title("Portfolio Correlation Heatmap")
st.caption(
    "วิเคราะห์ว่าหุ้นหรือสินทรัพย์ในพอร์ตเคลื่อนไหวคล้ายกันมากน้อยแค่ไหน "
    "โดยคำนวณ Correlation จาก Return ไม่ใช่จากระดับราคา"
)


DEFAULT_TICKERS = "NVDA, AAPL, MSFT, GOOGL, AMZN, META, AVGO, TSM, VOO, GLD"


def parse_tickers(raw_text: str) -> list[str]:
    """Parse comma/newline-separated tickers and remove duplicates."""
    normalized = raw_text.replace("\n", ",")
    tickers = [item.strip().upper() for item in normalized.split(",")]
    tickers = [ticker for ticker in tickers if ticker]

    unique_tickers: list[str] = []
    for ticker in tickers:
        if ticker not in unique_tickers:
            unique_tickers.append(ticker)

    if len(unique_tickers) < 2:
        raise ValueError("กรุณาระบุ ticker อย่างน้อย 2 ตัว")
    if len(unique_tickers) > 25:
        raise ValueError("รองรับสูงสุด 25 ticker ต่อครั้งเพื่อให้กราฟอ่านง่าย")

    return unique_tickers


@st.cache_data(ttl=900, show_spinner=False)
def download_prices(
    tickers: tuple[str, ...],
    start_date: date,
    end_date: date,
    interval: str,
    auto_adjust: bool,
) -> tuple[pd.DataFrame, list[str]]:
    """Download closing prices and return valid tickers plus failures."""
    if start_date >= end_date:
        raise ValueError("วันที่เริ่มต้นต้องอยู่ก่อนวันที่สิ้นสุด")

    end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)

    raw = yf.download(
        tickers=list(tickers),
        start=start_date.isoformat(),
        end=end_exclusive.date().isoformat(),
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
        threads=True,
    )

    if raw.empty:
        raise ValueError("ไม่พบข้อมูลในช่วงเวลาที่เลือก")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError("ข้อมูลที่ได้รับไม่มีคอลัมน์ Close")
        close = raw["Close"].copy()
    else:
        # Single-ticker fallback; normally prevented because minimum is two.
        if "Close" not in raw.columns:
            raise ValueError("ข้อมูลที่ได้รับไม่มีคอลัมน์ Close")
        close = raw[["Close"]].copy()
        close.columns = [tickers[0]]

    if isinstance(close, pd.Series):
        close = close.to_frame()

    close.index = pd.to_datetime(close.index)
    if isinstance(close.index, pd.DatetimeIndex) and close.index.tz is not None:
        close.index = close.index.tz_convert(None)

    close = close.sort_index()
    close = close.replace([np.inf, -np.inf], np.nan)

    valid_columns = [
        column for column in close.columns
        if close[column].notna().sum() >= 3
    ]
    failed_tickers = [ticker for ticker in tickers if ticker not in valid_columns]

    close = close[valid_columns]

    if close.shape[1] < 2:
        raise ValueError(
            "มี ticker ที่ดึงข้อมูลสำเร็จน้อยกว่า 2 ตัว "
            "โปรดตรวจสอบ ticker หรือเปลี่ยนช่วงเวลา"
        )

    close.index.name = "Date"
    return close, failed_tickers


def calculate_returns(
    prices: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    """Calculate arithmetic or log returns."""
    if method == "Log return":
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change(fill_method=None)

    return returns.replace([np.inf, -np.inf], np.nan)


def calculate_correlation(
    returns: pd.DataFrame,
    min_periods: int,
) -> pd.DataFrame:
    """Calculate Pearson correlation using overlapping observations."""
    correlation = returns.corr(method="pearson", min_periods=min_periods)

    if correlation.empty or correlation.shape[0] < 2:
        raise ValueError("ข้อมูลไม่เพียงพอสำหรับคำนวณ Correlation")

    return correlation


def classify_correlation(value: float) -> str:
    """Provide a simple descriptive label for absolute correlation."""
    absolute = abs(value)
    if absolute >= 0.80:
        return "สูงมาก"
    if absolute >= 0.60:
        return "สูง"
    if absolute >= 0.40:
        return "ปานกลาง"
    if absolute >= 0.20:
        return "ต่ำ"
    return "ต่ำมาก"


def pairwise_table(correlation: pd.DataFrame) -> pd.DataFrame:
    """Convert a correlation matrix into a ranked pairwise table."""
    rows = []
    for ticker_a, ticker_b in combinations(correlation.columns, 2):
        value = correlation.loc[ticker_a, ticker_b]
        if pd.notna(value):
            rows.append(
                {
                    "Asset A": ticker_a,
                    "Asset B": ticker_b,
                    "Correlation": float(value),
                    "Absolute Correlation": float(abs(value)),
                    "ระดับ": classify_correlation(float(value)),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "Asset A",
                "Asset B",
                "Correlation",
                "Absolute Correlation",
                "ระดับ",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values("Absolute Correlation", ascending=False)
        .reset_index(drop=True)
    )


def portfolio_summary(pair_table: pd.DataFrame) -> dict[str, object]:
    """Create simple summary statistics from pairwise correlations."""
    if pair_table.empty:
        return {
            "average": np.nan,
            "highest_pair": "-",
            "highest_value": np.nan,
            "lowest_pair": "-",
            "lowest_value": np.nan,
            "high_pair_count": 0,
        }

    highest = pair_table.iloc[0]
    lowest = pair_table.sort_values("Correlation", ascending=True).iloc[0]

    return {
        "average": float(pair_table["Correlation"].mean()),
        "highest_pair": f"{highest['Asset A']} / {highest['Asset B']}",
        "highest_value": float(highest["Correlation"]),
        "lowest_pair": f"{lowest['Asset A']} / {lowest['Asset B']}",
        "lowest_value": float(lowest["Correlation"]),
        "high_pair_count": int(
            (pair_table["Absolute Correlation"] >= 0.80).sum()
        ),
    }


def create_heatmap(
    correlation: pd.DataFrame,
    annotate: bool,
    mask_upper: bool,
    title: str,
):
    """Create the requested Seaborn correlation heatmap."""
    figure_size = max(8, min(18, 0.75 * len(correlation.columns) + 5))
    figure, axis = plt.subplots(figsize=(figure_size, figure_size * 0.78))

    mask = None
    if mask_upper:
        mask = np.triu(
            np.ones_like(correlation, dtype=bool),
            k=1,
        )

    sns.heatmap(
        correlation,
        mask=mask,
        annot=annotate,
        fmt=".2f",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson correlation"},
        ax=axis,
    )
    axis.set_title(title, pad=14)
    axis.set_xlabel("")
    axis.set_ylabel("")
    figure.tight_layout()
    return figure


def create_normalized_price_chart(prices: pd.DataFrame):
    """Normalize each asset to 100 for a comparable price chart."""
    normalized = prices.divide(prices.ffill().bfill().iloc[0]).multiply(100)
    figure, axis = plt.subplots(figsize=(14, 7))

    for column in normalized.columns:
        axis.plot(normalized.index, normalized[column], label=column)

    axis.set_title("Normalized Price Performance (Start = 100)")
    axis.set_ylabel("Indexed value")
    axis.grid(alpha=0.25)
    axis.legend(ncol=min(4, len(normalized.columns)))
    figure.autofmt_xdate()
    figure.tight_layout()
    return figure


def figure_to_png(figure) -> bytes:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=170, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


def format_correlation(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:.2f}"


with st.sidebar:
    st.header("ตั้งค่าการวิเคราะห์")

    with st.form("correlation_controls"):
        raw_tickers = st.text_area(
            "Ticker แยกด้วย comma หรือขึ้นบรรทัดใหม่",
            value=DEFAULT_TICKERS,
            height=150,
            help=(
                "ตัวอย่างหุ้นไทย: PTT.BK, DELTA.BK | "
                "ETF: VOO, QQQ, GLD | Crypto: BTC-USD"
            ),
        )

        today = date.today()
        selected_dates = st.date_input(
            "ช่วงวันที่",
            value=(today - timedelta(days=365 * 3), today),
            max_value=today,
        )

        interval = st.selectbox(
            "ความถี่ของข้อมูล",
            options=["1d", "1wk", "1mo"],
            format_func=lambda value: {
                "1d": "รายวัน",
                "1wk": "รายสัปดาห์",
                "1mo": "รายเดือน",
            }[value],
        )

        return_method = st.radio(
            "ประเภท Return",
            options=["Simple return", "Log return"],
            horizontal=True,
        )

        auto_adjust = st.toggle(
            "ปรับราคาสำหรับ Split/Dividend",
            value=True,
        )

        min_periods = st.number_input(
            "จำนวนข้อมูลซ้อนทับขั้นต่ำ",
            min_value=3,
            max_value=500,
            value=30,
            step=1,
            help=(
                "คู่สินทรัพย์ต้องมี Return ที่ไม่ว่างซ้อนทับกันอย่างน้อย "
                "ตามจำนวนนี้จึงจะคำนวณ Correlation"
            ),
        )

        annotate = st.checkbox(
            "แสดงค่าตัวเลขใน Heatmap",
            value=True,
        )
        mask_upper = st.checkbox(
            "ซ่อนครึ่งบนของ Matrix",
            value=False,
        )

        st.form_submit_button(
            "วิเคราะห์ Correlation",
            use_container_width=True,
        )

    st.caption(
        "Correlation ควรคำนวณจาก Return ไม่ใช่ราคาดิบ "
        "และผลลัพธ์ขึ้นกับช่วงเวลาที่เลือก"
    )


if not isinstance(selected_dates, (tuple, list)) or len(selected_dates) != 2:
    st.warning("กรุณาเลือกทั้งวันที่เริ่มต้นและวันที่สิ้นสุด")
    st.stop()

start_date, end_date = selected_dates

try:
    tickers = parse_tickers(raw_tickers)
except ValueError as error:
    st.error(str(error))
    st.stop()

try:
    with st.spinner("กำลังโหลดราคาและคำนวณ Correlation..."):
        prices, failed_tickers = download_prices(
            tickers=tuple(tickers),
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            auto_adjust=auto_adjust,
        )
        returns = calculate_returns(prices, return_method)
        correlation = calculate_correlation(
            returns=returns,
            min_periods=int(min_periods),
        )
        pairs = pairwise_table(correlation)
        summary = portfolio_summary(pairs)
except Exception as error:
    st.error(f"ไม่สามารถวิเคราะห์ข้อมูลได้: {error}")
    st.stop()

if failed_tickers:
    st.warning(
        "ไม่พบข้อมูลหรือข้อมูลไม่เพียงพอสำหรับ: "
        + ", ".join(failed_tickers)
    )

st.subheader(
    f"{len(correlation.columns)} สินทรัพย์ · "
    f"{start_date:%d %b %Y} – {end_date:%d %b %Y}"
)
st.caption(
    f"Price observations: {len(prices):,} | "
    f"Return observations: {len(returns.dropna(how='all')):,} | "
    f"Interval: {interval}"
)

metric_columns = st.columns(4)
metric_columns[0].metric(
    "Correlation เฉลี่ย",
    format_correlation(summary["average"]),
)
metric_columns[1].metric(
    "คู่สัมพันธ์สูงสุด",
    summary["highest_pair"],
    format_correlation(summary["highest_value"]),
)
metric_columns[2].metric(
    "คู่สัมพันธ์ต่ำสุด",
    summary["lowest_pair"],
    format_correlation(summary["lowest_value"]),
)
metric_columns[3].metric(
    "คู่ |Corr| ≥ 0.80",
    f"{summary['high_pair_count']:,}",
)

heatmap_tab, pairs_tab, prices_tab, data_tab, learn_tab = st.tabs(
    [
        "Correlation Heatmap",
        "คู่สินทรัพย์",
        "ราคาเปรียบเทียบ",
        "ข้อมูล",
        "สรุปการเรียนรู้",
    ]
)

with heatmap_tab:
    heatmap_figure = create_heatmap(
        correlation=correlation,
        annotate=annotate,
        mask_upper=mask_upper,
        title=f"Return Correlation Heatmap ({interval})",
    )
    st.pyplot(heatmap_figure, use_container_width=True)

    heatmap_png = figure_to_png(heatmap_figure)
    st.download_button(
        "ดาวน์โหลด Heatmap PNG",
        data=heatmap_png,
        file_name="correlation_heatmap.png",
        mime="image/png",
    )
    plt.close(heatmap_figure)

    st.info(
        "ค่าใกล้ +1 หมายถึงเคลื่อนไหวไปทางเดียวกันมาก "
        "ค่าใกล้ 0 หมายถึงไม่มีความสัมพันธ์เชิงเส้นชัดเจน "
        "และค่าใกล้ -1 หมายถึงมักเคลื่อนไหวสวนทางกัน"
    )

with pairs_tab:
    if pairs.empty:
        st.warning("ไม่มีคู่สินทรัพย์ที่มีข้อมูลเพียงพอ")
    else:
        threshold = st.slider(
            "แสดงคู่ที่มี |Correlation| ตั้งแต่",
            min_value=0.0,
            max_value=1.0,
            value=0.60,
            step=0.05,
        )

        filtered_pairs = pairs[
            pairs["Absolute Correlation"] >= threshold
        ].copy()

        st.dataframe(
            filtered_pairs,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Correlation": st.column_config.NumberColumn(
                    format="%.3f"
                ),
                "Absolute Correlation": st.column_config.NumberColumn(
                    format="%.3f"
                ),
            },
        )

        pairs_csv = pairs.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "ดาวน์โหลดตารางคู่สินทรัพย์ CSV",
            data=pairs_csv,
            file_name="pairwise_correlations.csv",
            mime="text/csv",
        )

with prices_tab:
    normalized_figure = create_normalized_price_chart(prices)
    st.pyplot(normalized_figure, use_container_width=True)
    plt.close(normalized_figure)

    st.caption(
        "กราฟนี้ Normalize ราคาเริ่มต้นเป็น 100 เพื่อเปรียบเทียบเส้นทางราคา "
        "แต่ Correlation ด้านบนคำนวณจาก Return"
    )

with data_tab:
    correlation_csv = correlation.to_csv().encode("utf-8-sig")
    returns_csv = returns.reset_index().to_csv(index=False).encode("utf-8-sig")
    prices_csv = prices.reset_index().to_csv(index=False).encode("utf-8-sig")

    st.markdown("### Correlation Matrix")
    st.dataframe(correlation, use_container_width=True)

    st.download_button(
        "ดาวน์โหลด Correlation Matrix",
        data=correlation_csv,
        file_name="correlation_matrix.csv",
        mime="text/csv",
    )
    st.download_button(
        "ดาวน์โหลด Return Data",
        data=returns_csv,
        file_name="asset_returns.csv",
        mime="text/csv",
    )
    st.download_button(
        "ดาวน์โหลด Price Data",
        data=prices_csv,
        file_name="asset_prices.csv",
        mime="text/csv",
    )

with learn_tab:
    st.markdown(
        """
        ### สิ่งที่โปรเจกต์นี้ฝึก

        - ดึงข้อมูลหลายสินทรัพย์ด้วย `yfinance`
        - จัดการ Missing Values และวันที่ด้วย `pandas`
        - คำนวณ Simple Return และ Log Return
        - สร้าง Pearson Correlation Matrix
        - แปลง Matrix เป็นตารางคู่สินทรัพย์
        - สร้าง Heatmap ด้วย `seaborn`
        - ออกแบบ Dashboard และ Export ข้อมูลด้วย Streamlit
        """
    )

    st.markdown("### การตีความเพื่อ Diversification")
    st.write(
        "- คู่ที่ Correlation สูงมากอาจให้การกระจายความเสี่ยงน้อยกว่าที่คิด"
    )
    st.write(
        "- สินทรัพย์ที่ Correlation ต่ำหรือเป็นลบอาจช่วยลดการเคลื่อนไหวพร้อมกันของพอร์ต"
    )
    st.write(
        "- Correlation ไม่ได้วัดความเสี่ยงทั้งหมด และสามารถเปลี่ยนได้ตามช่วงตลาด"
    )
    st.write(
        "- ควรพิจารณา Volatility, Drawdown, สัดส่วนพอร์ต และปัจจัยพื้นฐานร่วมด้วย"
    )

    st.warning(
        "Correlation ไม่ได้แปลว่าเหตุและผล และค่าที่เห็นจากข้อมูลย้อนหลัง "
        "ไม่ได้รับประกันความสัมพันธ์ในอนาคต"
    )

st.divider()
st.caption(
    "เพื่อการศึกษาเท่านั้น ไม่ใช่คำแนะนำการลงทุน "
    "ข้อมูลจาก yfinance อาจล่าช้า ไม่ครบถ้วน หรือแตกต่างจากข้อมูลตลาดทางการ"
)
