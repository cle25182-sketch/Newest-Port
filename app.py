import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
from datetime import datetime, timedelta
import streamlit as st

st.set_page_config(
    page_title="QuantLab — ระบบจำลองและจัดพอร์ตหุ้นเชิงปริมาณ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

MAX_TICKERS_ALLOWED = 10  # จำกัดจำนวนหุ้นสูงสุดเพื่อความรวดเร็วในการประมวลผล

# Custom CSS Styling
st.markdown("""
<style>
    .disclaimer-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
    }
    .concept-box {
        background-color: #0f172a;
        border-left: 4px solid #38bdf8;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 10px 0px;
    }
</style>
""", unsafe_allow_html=True)

STRAT_DISPLAY_NAMES = {
    "equal": "A: Equal-Weight",
    "markowitz": "B: Markowitz",
    "mcap": "C: Market-Cap",
    "shrinkage": "D: Shrinkage"
}

STRAT_FULL_NAMES = {
    "equal": "A: Equal-weight (สัดส่วนเท่ากัน)",
    "markowitz": "B: Markowitz (สมการคณิตศาสตร์)",
    "mcap": "C: Market-cap (ตามมูลค่าบริษัท)",
    "shrinkage": "D: Markowitz-Shrinkage (ลดสัญญาณรบกวน)"
}


def get_currency_info(tickers):
    """ตรวจสอบสกุลเงินอัตโนมัติจาก Ticker Suffix"""
    if not tickers:
        return "THB", "บาท"
    if all(t.endswith(".BK") for t in tickers):
        return "THB", "บาท"
    if all(t.endswith(".T") for t in tickers):
        return "JPY", "เยน"
    if all(t.endswith(".HK") for t in tickers):
        return "HKD", "ดอลลาร์ฮ่องกง"
    if all(not t.endswith((".BK", ".T", ".HK")) for t in tickers):
        return "USD", "ดอลลาร์สหรัฐ"
    return "USD / Mixed", "ดอลลาร์/สกุลเงินผสม"


@st.cache_data(ttl=43200, show_spinner=False)
def load_price_data(tickers, years_back):
    """ดึงข้อมูลราคาหุ้นย้อนหลัง พร้อมระบบจัดการ Error และตัด Timezone"""
    import yfinance as yf
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=years_back * 365)).strftime("%Y-%m-%d")
    
    try:
        raw = yf.download(list(tickers), start=start_date, end=end_date, progress=False, auto_adjust=True)["Close"]
        if raw.empty:
            return None, [], "ไม่พบข้อมูลราคาหุ้นสำหรับรหัสที่ระบุ"
        if isinstance(raw, pd.Series):
            raw = raw.to_frame(tickers[0])
            
        if hasattr(raw.index, "tz") and raw.index.tz is not None:
            raw.index = raw.index.tz_localize(None)
            
        raw = raw.ffill()
        valid = [t for t in tickers if t in raw.columns and raw[t].notna().sum() >= 30]
        return raw[valid].dropna(), valid, None
    except Exception as e:
        return None, [], f"ไม่สามารถเชื่อมต่อกับ Yahoo Finance ได้ในขณะนี้ ({str(e)})"


@st.cache_data(ttl=43200, show_spinner=False)
def load_benchmark(bench_choice, tickers, years_back):
    """ดึงข้อมูล Benchmark พร้อมรองรับตลาดหุ้นไทย สหรัฐ ญี่ปุ่น ฮ่องกง และพอร์ตผสม"""
    import yfinance as yf
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=years_back * 365)).strftime("%Y-%m-%d")
    
    symbol_map = {
        "🇹🇭 SET Index (หุ้นไทย)": (["^SET.BK", "TDEX.BK"], "SET Index"),
        "🇺🇸 S&P 500 (หุ้นใหญ่สหรัฐฯ)": (["^GSPC", "SPY"], "S&P 500"),
        "🚀 Nasdaq 100 (หุ้นเทคฯ สหรัฐฯ)": (["^NDX", "QQQ"], "Nasdaq 100"),
        "🇯🇵 Nikkei 225 (หุ้นญี่ปุ่น)": (["^N225", "EWJ"], "Nikkei 225"),
        "🇭🇰 Hang Seng Index (หุ้นฮ่องกง)": (["^HSI", "2800.HK"], "Hang Seng"),
        "🌐 MSCI ACWI (หุ้นทั่วโลก / พอร์ตผสม)": (["ACWI", "URTH"], "MSCI ACWI"),
    }
    
    if bench_choice == "⚡ อัตโนมัติ (Auto-detect)":
        if all(t.endswith(".BK") for t in tickers):
            symbols, label = ["^SET.BK", "TDEX.BK"], "SET Index"
        elif all(t.endswith(".T") for t in tickers):
            symbols, label = ["^N225", "EWJ"], "Nikkei 225"
        elif all(t.endswith(".HK") for t in tickers):
            symbols, label = ["^HSI", "2800.HK"], "Hang Seng"
        elif all(not t.endswith((".BK", ".T", ".HK")) for t in tickers):
            symbols, label = ["^GSPC", "SPY"], "S&P 500"
        else:
            symbols, label = ["ACWI", "URTH", "^GSPC"], "MSCI ACWI"
    else:
        symbols, label = symbol_map.get(bench_choice, (["^SET.BK", "TDEX.BK"], "Benchmark"))

    for sym in symbols:
        try:
            raw = yf.download(sym, start=start_date, end=end_date, progress=False)
            if raw.empty:
                continue
            
            if isinstance(raw.columns, pd.MultiIndex):
                if "Close" in raw.columns.levels[0]:
                    series = raw["Close"].iloc[:, 0]
                elif "Adj Close" in raw.columns.levels[0]:
                    series = raw["Adj Close"].iloc[:, 0]
                else:
                    series = raw.iloc[:, 0]
            else:
                if "Close" in raw.columns:
                    series = raw["Close"]
                elif "Adj Close" in raw.columns:
                    series = raw["Adj Close"]
                else:
                    series = raw.iloc[:, 0]
                    
            series = series.dropna()
            if len(series) > 10:
                if hasattr(series.index, "tz") and series.index.tz is not None:
                    series.index = series.index.tz_localize(None)
                return series, sym, label
        except Exception:
            continue
            
    return pd.Series(dtype=float), symbols[0], label


@st.cache_data(ttl=43200, show_spinner=False)
def load_market_caps(tickers):
    import yfinance as yf
    caps = {}
    for t in tickers:
        try:
            t_obj = yf.Ticker(t)
            info = t_obj.info
            shares = info.get("sharesOutstanding")
            if shares is None:
                mc = info.get("marketCap")
                price = info.get("regularMarketPrice") or info.get("previousClose")
                if mc and price:
                    shares = mc / price
            caps[t] = shares
        except Exception:
            caps[t] = None
    return caps


def _clean_shares(v):
    if v is None:
        return None
    try:
        return None if np.isnan(v) else v
    except TypeError:
        return v


def ledoit_wolf_shrinkage(X):
    """Ledoit-Wolf Covariance Shrinkage Matrix Calculation"""
    T, N = X.shape
    Xc = X - X.mean(axis=0)
    S = (Xc.T @ Xc) / T
    mu = np.trace(S) / N
    F = mu * np.eye(N)

    outer_all = np.einsum("ti,tj->tij", Xc, Xc)
    pi_mat = ((outer_all - S) ** 2).mean(axis=0)
    pi_hat = pi_mat.sum()
    rho_hat = np.trace(pi_mat)
    gamma_hat = np.sum((S - F) ** 2)

    kappa_hat = (pi_hat - rho_hat) / gamma_hat if gamma_hat > 1e-18 else 0.0
    delta = max(0.0, min(1.0, kappa_hat / T))

    shrunk_cov = delta * F + (1 - delta) * S
    return shrunk_cov, delta


def get_weights(train_ret, train_last_price, shares_arr, n, use_marketcap, rf_rate):
    """คำนวณน้ำหนักพอร์ตสำหรับทั้ง 4 กลยุทธ์"""
    mu, cov = train_ret.mean(), train_ret.cov()
    w_equal = np.array([1 / n] * n)

    def neg_sharpe(w, mu_arr, cov_arr):
        ret = np.sum(mu_arr * w) * 252 - rf_rate
        vol = np.sqrt(max(w @ cov_arr @ w, 0)) * np.sqrt(252)
        return -ret / vol if vol > 1e-10 else 1e6

    bounds = tuple((0, 1) for _ in range(n))
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    # Strategy B: Standard Markowitz
    opt = minimize(neg_sharpe, w_equal, args=(mu.values, cov.values),
                   method="SLSQP", bounds=bounds, constraints=constraints)
    w_markowitz = opt.x if opt.success else w_equal

    # Strategy D: Markowitz-Shrinkage
    shrunk_cov, delta = ledoit_wolf_shrinkage(train_ret.values)
    opt_shrink = minimize(neg_sharpe, w_equal, args=(mu.values, shrunk_cov),
                           method="SLSQP", bounds=bounds, constraints=constraints)
    w_shrink = opt_shrink.x if opt_shrink.success else w_equal

    weights = {
        "equal": w_equal,
        "markowitz": w_markowitz,
        "shrinkage": w_shrink,
    }
    if use_marketcap and shares_arr is not None and len(shares_arr) == n:
        mcap = train_last_price * shares_arr
        if mcap.sum() > 0:
            weights["mcap"] = mcap / mcap.sum()
        else:
            weights["mcap"] = w_equal
    return weights, (opt.success and opt_shrink.success), delta


def block_bootstrap_pvalue(diff_values, block_size=4, n_boot=2000, seed=0):
    """ทดสอบว่าค่าเฉลี่ยผลต่างต่างจาก 0 จริงไหม ด้วย moving block bootstrap -- ทนทานต่อการที่แต่ละรอบ
    ไม่เป็นอิสระจากกันสมบูรณ์ (เพราะช่วง train ของรอบที่ติดกันทับซ้อนกันถึง 75%)"""
    rng = np.random.default_rng(seed)
    n = len(diff_values)
    if n < block_size * 2:
        return np.nan
    observed_mean = diff_values.mean()
    n_blocks = int(np.ceil(n / block_size))
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = []
        for _ in range(n_blocks):
            start = rng.integers(0, n - block_size + 1)
            idx.extend(range(start, start + block_size))
        boot_means[b] = diff_values[idx[:n]].mean()
    boot_centered = boot_means - boot_means.mean()
    return np.mean(np.abs(boot_centered) >= abs(observed_mean))


def evaluate(w, test_ret, rf_rate):
    port_ret = test_ret.values @ w
    cum = np.cumprod(1 + port_ret)
    ann_ret = port_ret.mean() * 252
    ann_vol = port_ret.std() * np.sqrt(252)
    sharpe = (ann_ret - rf_rate) / ann_vol if ann_vol > 1e-10 else np.nan
    running_max = np.maximum.accumulate(cum)
    max_dd = ((cum - running_max) / running_max).min()
    return cum[-1] - 1, ann_ret, ann_vol, sharpe, max_dd


def run_walk_forward(data, shares_arr, use_marketcap, train_window, test_window, rf_rate):
    all_returns = data.pct_change().dropna()
    n = data.shape[1]
    records, failed = [], 0
    daily_returns = {}
    weight_history = {}
    last_weights = {}
    deltas = []
    start, round_num = 0, 0
    while start + train_window + test_window <= len(all_returns):
        round_num += 1
        train_ret = all_returns.iloc[start: start + train_window]
        test_ret = all_returns.iloc[start + train_window: start + train_window + test_window]
        train_last_price = data.iloc[start + train_window].values
        weights, ok, delta = get_weights(train_ret, train_last_price, shares_arr, n, use_marketcap, rf_rate)
        failed += 0 if ok else 1
        deltas.append(delta)
        last_weights = weights
        for key, w in weights.items():
            weight_history.setdefault(key, []).append({"round": round_num, **{t: w[i] for i, t in enumerate(data.columns)}})
            total, ar, av, sh, mdd = evaluate(w, test_ret, rf_rate)
            records.append({
                "round": round_num,
                "strategy": key,
                "total_return": total,
                "ann_return": ar,
                "ann_vol": av,
                "sharpe": sh,
                "max_drawdown": mdd
            })
            port_ret_series = pd.Series(test_ret.values @ w, index=test_ret.index)
            daily_returns.setdefault(key, []).append(port_ret_series)
        start += test_window
    daily_returns = {k: pd.concat(v) for k, v in daily_returns.items()}
    weight_history = {k: pd.DataFrame(v).set_index("round") for k, v in weight_history.items()}
    avg_delta = float(np.mean(deltas)) if deltas else 0.0
    return pd.DataFrame(records), round_num, failed, daily_returns, last_weights, weight_history, avg_delta, all_returns


def cumulative_growth_with_turnover(daily_returns, weight_history, capital, cost_pct, test_window):
    curves = {}
    for key, ret_series in daily_returns.items():
        equity, values = capital, []
        w_df = weight_history.get(key)
        
        for i, (date, r) in enumerate(ret_series.items()):
            if i > 0 and i % test_window == 0 and w_df is not None:
                round_idx = (i // test_window) + 1
                if round_idx in w_df.index and (round_idx - 1) in w_df.index:
                    w_old = w_df.loc[round_idx - 1].values
                    w_new = w_df.loc[round_idx].values
                    turnover = np.sum(np.abs(w_new - w_old)) / 2.0
                    rebalance_cost = equity * turnover * cost_pct
                    equity -= rebalance_cost
            
            equity *= (1 + r)
            values.append(equity)
        curves[key] = pd.Series(values, index=ret_series.index)
    return curves


def create_share_card(best_key, annual_ret, sharpe, max_dd, tickers, capital, curr_name):
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    ax.axis('off')

    ax.text(0.05, 0.88, "QUANTLAB PORTFOLIO SUMMARY", fontsize=15, fontweight='bold', color='#38bdf8')
    ticker_str = ", ".join(tickers[:6]) + ("..." if len(tickers) > 6 else "")
    ax.text(0.05, 0.80, f"Assets: {ticker_str}", fontsize=9, color='#94a3b8')

    ax.text(0.05, 0.66, "BEST STRATEGY", fontsize=9, fontweight='bold', color='#cbd5e1')
    strat_clean = STRAT_DISPLAY_NAMES.get(best_key, best_key)
    ax.text(0.05, 0.54, strat_clean, fontsize=16, fontweight='bold', color='#4ade80')

    ax.text(0.05, 0.35, "ANNUAL RETURN", fontsize=8, color='#94a3b8')
    ax.text(0.05, 0.22, f"{annual_ret*100:+.1f}%", fontsize=15, fontweight='bold', color='white')

    ax.text(0.38, 0.35, "SHARPE RATIO", fontsize=8, color='#94a3b8')
    ax.text(0.38, 0.22, f"{sharpe:.2f}", fontsize=15, fontweight='bold', color='#facc15')

    ax.text(0.70, 0.35, "MAX DRAWDOWN", fontsize=8, color='#94a3b8')
    ax.text(0.70, 0.22, f"{max_dd*100:.1f}%", fontsize=15, fontweight='bold', color='#f87171')

    ax.text(0.05, 0.08, f"Initial Capital: {capital:,.0f} {curr_name}  |  Generated by QuantLab Engine", fontsize=8, color='#64748b')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


st.title("📈 QuantLab — ระบบจำลองจัดพอร์ตหุ้นเชิงปริมาณ")
st.caption("ระบบจำลองพอร์ตการลงทุน ทดสอบด้วยวิธี Walk-Forward Validation")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ ตั้งค่าการวิเคราะห์")
    
    if "ticker_text" not in st.session_state:
        st.session_state.ticker_text = "PTT.BK, CPALL.BK, AOT.BK, KBANK.BK, ADVANC.BK"

    ticker_input = st.text_input(
        "พิมพ์รหัสหุ้น (คั่นด้วยจุลภาค ,)",
        key="ticker_text",
        help="ใส่รหัสหุ้น Yahoo Finance เช่น PTT.BK, AAPL, 7203.T, 0700.HK"
    )
    selected_raw = list(dict.fromkeys([t.strip().upper() for t in ticker_input.split(",") if t.strip()]))

    if len(selected_raw) > MAX_TICKERS_ALLOWED:
        st.warning(f"⚠️ ระบบจำกัดสูงสุดที่ {MAX_TICKERS_ALLOWED} หุ้น (เลือก {MAX_TICKERS_ALLOWED} ตัวแรกให้)")
        selected = selected_raw[:MAX_TICKERS_ALLOWED]
    else:
        selected = selected_raw

    # Dynamic Currency Detection
    curr_code, curr_name = get_currency_info(selected)
    capital = st.number_input(f"เงินลงทุนเริ่มต้น ({curr_name} / {curr_code})", min_value=1000, value=100000, step=5000)

    benchmark_option = st.selectbox(
        "ตัวเปรียบเทียบดัชนี (Benchmark)",
        [
            "⚡ อัตโนมัติ (Auto-detect)",
            "🇹🇭 SET Index (หุ้นไทย)",
            "🇺🇸 S&P 500 (หุ้นใหญ่สหรัฐฯ)",
            "🚀 Nasdaq 100 (หุ้นเทคฯ สหรัฐฯ)",
            "🇯🇵 Nikkei 225 (หุ้นญี่ปุ่น)",
            "🇭🇰 Hang Seng Index (หุ้นฮ่องกง)",
            "🌐 MSCI ACWI (หุ้นทั่วโลก / พอร์ตผสม)"
        ],
        help="เลือกดัชนีอ้างอิงเพื่อเปรียบเทียบผลตอบแทนกับพอร์ตของคุณ"
    )

    with st.expander("ตั้งค่าการทดสอบขั้นสูง"):
        rf_rate_pct = st.slider("อัตราดอกเบี้ยไร้ความเสี่ยง Rf (%/ปี)", 0.0, 5.0, 2.0, step=0.25) / 100
        train_window = st.slider("ช่วง Train ข้อมูล (วัน)", 126, 378, 252, step=21)
        test_window = st.slider("ช่วง Test ต่อรอบ (วัน)", 21, 126, 63, step=21)
        years_back = st.slider("ข้อมูลย้อนหลัง (ปี)", 3, 25, 5)
        cost_pct = st.slider("ค่าธรรมเนียมซื้อขายปรับพอร์ต (%)", 0.0, 1.0, 0.1, step=0.05) / 100
        stress_option = st.selectbox(
            "จำลองวิกฤตเฉพาะช่วง",
            ["ไม่ระบุ", "COVID-19 (ก.พ.–เม.ย. 2020)", "เงินเฟ้อ/ดอกเบี้ยขาขึ้น (2022)"]
        )
        STRESS_RANGES = {
            "COVID-19 (ก.พ.–เม.ย. 2020)": ("2020-02-01", "2020-04-30"),
            "เงินเฟ้อ/ดอกเบี้ยขาขึ้น (2022)": ("2022-01-01", "2022-12-31"),
        }

    run = st.button("เริ่มวิเคราะห์", type="primary", use_container_width=True)

if not run:
    st.info("👈 ปรับแต่งตัวเลือกทางซ้ายมือ แล้วกด **เริ่มวิเคราะห์** เพื่อประมวลผล")
    st.stop()

if len(selected) < 2:
    st.error("กรุณาใส่รหัสหุ้นอย่างน้อย 2 ตัวขึ้นไปเพื่อจัดพอร์ตกระจายความเสี่ยง")
    st.stop()

with st.spinner("กำลังดึงราคาหุ้นและประมวลผลทางสถิติ..."):
    data, valid_tickers, error_msg = load_price_data(tuple(selected), years_back)
    
    if error_msg:
        st.error(f"⚠️ การดึงข้อมูลล้มเหลว: {error_msg}")
        st.stop()

selected = valid_tickers
if len(selected) < 2:
    st.error("เหลือหุ้นที่ข้อมูลสมบูรณ์น้อยกว่า 2 ตัว กรุณาเปลี่ยนรหัสหุ้น")
    st.stop()

caps_raw = load_market_caps(tuple(selected))
clean_caps = {t: _clean_shares(caps_raw.get(t)) for t in selected}
valid_shares = [v for v in clean_caps.values() if v is not None and v > 0]

if valid_shares:
    median_shares = np.median(valid_shares)
    for t in selected:
        if clean_caps[t] is None:
            clean_caps[t] = median_shares
    use_marketcap = True
    shares_arr = np.array([clean_caps[t] for t in selected])
else:
    use_marketcap = False
    shares_arr = None

df, n_folds, n_failed, daily_returns, last_weights, weight_history, avg_delta, all_returns_full = run_walk_forward(
    data, shares_arr, use_marketcap, train_window, test_window, rf_rate_pct
)

if df.empty:
    st.error("ข้อมูลย้อนหลังมีไม่เพียงพอต่อการแบ่งรอบทดสอบ ลองลดช่วงวัน Train/Test ในเมนูขั้นสูง")
    st.stop()

strategies = list(df["strategy"].unique())
summary = df.groupby("strategy")[["ann_return", "ann_vol", "sharpe", "max_drawdown"]].mean().reindex(strategies)
best_key = summary["sharpe"].idxmax()
best_strategy = STRAT_FULL_NAMES.get(best_key, best_key)
best_return = summary.loc[best_key, "ann_return"]
best_vol = summary.loc[best_key, "ann_vol"]
best_sharpe = summary.loc[best_key, "sharpe"]
best_mdd = summary.loc[best_key, "max_drawdown"]

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 สรุป", 
    "📈 เติบโต", 
    "🔬 สถิติเชิงลึก", 
    "📣 แชร์",
    "ℹ️ ข้อมูล"
])

# TAB 1: สรุปผล
wide = df.pivot(index="round", columns="strategy", values="sharpe")
significantly_better_than_all = True
for other in strategies:
    if other == best_key:
        continue
    _, p_val = stats.ttest_rel(wide[best_key], wide[other])
    if p_val >= 0.05:
        significantly_better_than_all = False
        break

with tab1:
    st.subheader(f"สรุปผล: กลยุทธ์ที่ Sharpe ดีที่สุดในการทดสอบนี้ ({curr_code})")
    if n_failed > 0:
        st.warning(f"⚠️ มีการคำนวณ Markowitz ไม่ลู่เข้าจำนวน {n_failed}/{n_folds} รอบ (ระบบใช้ Equal-Weight สำรองในรอบดังกล่าว)")

    if significantly_better_than_all:
        st.success(
            f"**{best_strategy}** ให้ Sharpe ratio ดีที่สุด และแตกต่างจากกลยุทธ์อื่นทุกตัวอย่างมีนัยสำคัญทางสถิติ "
            f"(p < 0.05) ในการทดสอบ {n_folds} รอบนี้"
        )
    else:
        st.info(
            f"**{best_strategy}** ให้ Sharpe ratio เฉลี่ยสูงสุดในการทดสอบนี้ แต่ยัง**ไม่ต่างจากกลยุทธ์อื่นอย่างมีนัยสำคัญทางสถิติ** "
            f"(ดูตาราง p-value ในแท็บ 'สถิติเชิงลึก') — ควรตีความว่า \"ยังแยกไม่ออกชัดเจนว่าวิธีไหนดีกว่าจริง\" มากกว่าฟันธงว่าตัวนี้ชนะ"
        )

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("ผลตอบแทนเฉลี่ยต่อปี", f"{best_return * 100:.1f}%")
    with col_m2:
        st.metric("ความผันผวนต่อปี", f"{best_vol * 100:.1f}%")
    with col_m3:
        st.metric("Sharpe ratio", f"{best_sharpe:.2f}")
    with col_m4:
        st.metric("Max Drawdown", f"{best_mdd * 100:.1f}%")

    st.markdown(f"#### ตัวอย่างการจัดสรรเงินตามกลยุทธ์ชนะเลิศ: {best_strategy}")
    winner_weights = last_weights[best_key]
    action_plan = []
    for ticker, weight in zip(selected, winner_weights):
        action_plan.append({
            "รหัสหุ้น": ticker,
            "สัดส่วน (%)": f"{weight * 100:.2f}%",
            f"จำนวนเงิน ({curr_code})": f"{capital * weight:,.2f}",
        })
    st.dataframe(pd.DataFrame(action_plan), use_container_width=True, hide_index=True)

    st.markdown("#### ตารางเปรียบเทียบทุกกลยุทธ์ (ค่าเฉลี่ยตลอดทุกรอบ)")
    display_summary = summary.copy()
    display_summary.index = [STRAT_FULL_NAMES.get(k, k) for k in display_summary.index]
    display_summary["ann_return"] = (display_summary["ann_return"] * 100).round(1).astype(str) + "%"
    display_summary["ann_vol"] = (display_summary["ann_vol"] * 100).round(1).astype(str) + "%"
    display_summary["max_drawdown"] = (display_summary["max_drawdown"] * 100).round(1).astype(str) + "%"
    display_summary["sharpe"] = display_summary["sharpe"].round(2)
    display_summary.columns = ["ผลตอบแทน/ปี", "ความผันผวน/ปี", "Sharpe ratio", "Max Drawdown"]
    st.dataframe(display_summary, use_container_width=True)

# TAB 2: การเติบโตของเงินทุน
with tab2:
    st.subheader(f"เงินลงทุนสะสม ({curr_code} — คิดทบต้นจริง + ค่าธรรมเนียม Rebalance)")
    curves = cumulative_growth_with_turnover(daily_returns, weight_history, capital, cost_pct, test_window)
    bench_raw, bench_used_symbol, bench_label = load_benchmark(benchmark_option, tuple(selected), years_back)

    fig_cum, ax_cum = plt.subplots(figsize=(10, 4.5), facecolor='#0f172a')
    ax_cum.set_facecolor('#1e293b')
    colors = ['#38bdf8', '#facc15', '#4ade80', '#c084fc']
    
    for i, (key, curve) in enumerate(curves.items()):
        lbl = STRAT_DISPLAY_NAMES.get(key, key)
        ax_cum.plot(curve.index, curve.values, label=lbl, color=colors[i % len(colors)], linewidth=1.6)

    if not bench_raw.empty:
        combined_index = next(iter(curves.values())).index
        if hasattr(combined_index, "tz") and combined_index.tz is not None:
            combined_index = combined_index.tz_localize(None)
            
        bench_aligned = bench_raw.reindex(bench_raw.index.union(combined_index)).ffill().reindex(combined_index)
        if bench_aligned.notna().sum() > 10:
            bench_ret = bench_aligned.pct_change().fillna(0)
            bench_curve = capital * (1 + bench_ret).cumprod()
            ax_cum.plot(bench_curve.index, bench_curve.values, label=f"{bench_label} ({bench_used_symbol})", linewidth=1.8,
                        linestyle="--", color="white")

    ax_cum.axhline(capital, color="gray", linewidth=0.7, linestyle=":")
    ax_cum.set_xlabel("Date", color='#94a3b8')
    ax_cum.set_ylabel(f"Portfolio Value ({curr_code})", color='#94a3b8')
    ax_cum.tick_params(colors='#cbd5e1')
    ax_cum.grid(True, linestyle='--', alpha=0.2)
    ax_cum.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='white', loc="upper left", fontsize=9)
    st.pyplot(fig_cum)

    if stress_option != "ไม่ระบุ":
        st.markdown(f"#### ทดสอบเฉพาะช่วงวิกฤต: {stress_option}")
        s_start, s_end = STRESS_RANGES[stress_option]
        stress_returns = all_returns_full.loc[s_start:s_end]
        if len(stress_returns) < 5:
            st.warning("ข้อมูลย้อนหลังที่มีไม่ครอบคลุมช่วงนี้ — ลองเพิ่ม 'ข้อมูลย้อนหลังกี่ปี' ในตั้งค่าขั้นสูง")
        else:
            stress_rows = []
            for key, w in last_weights.items():
                total, ar, av, sh, mdd = evaluate(w, stress_returns, rf_rate_pct)
                stress_rows.append({
                    "กลยุทธ์": STRAT_FULL_NAMES.get(key, key),
                    "ผลตอบแทนรวมช่วงนี้": f"{total:.1%}",
                    f"Max Drawdown ช่วงนี้": f"{mdd:.1%}",
                })
            st.dataframe(pd.DataFrame(stress_rows), use_container_width=True, hide_index=True)
            st.caption("ใช้สัดส่วนน้ำหนักจากรอบล่าสุดของแต่ละกลยุทธ์ ประเมินย้อนกลับเฉพาะช่วงวิกฤตที่เลือก (ไม่ใช่ walk-forward เต็มรูปแบบ เพราะช่วงสั้นเกินจะแบ่ง train/test ได้)")

# TAB 3: วิเคราะห์สถิติเชิงลึก
with tab3:
    st.subheader("🔬 วิเคราะห์สถิติเชิงลึกและประสิทธิภาพของกลยุทธ์")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("#### ค่าเฉลี่ยความเข้มข้นของการลดสัญญาณรบกวน (Ledoit-Wolf Delta)")
        st.info(rf"ค่าเฉลี่ย Delta ($\delta$) ตลอดการทดสอบคือ **{avg_delta:.4f}**" + "\n\n" +
                r"*หมายเหตุ: หากค่า $\delta$ เข้าใกล้ 1 แปลว่า Covariance Matrix ดั้งเดิมมีสัญญาณรบกวนสูงมาก ระบบจึงให้น้ำหนักกับการปรับลดความผันผวน (Shrinkage Target) มากขึ้น*")
    
    with col_s2:
        st.markdown("#### สรุปผลความเสี่ยงสูงสุด (Max Drawdown เปรียบเทียบ)")
        mdd_summary = df.groupby("strategy")["max_drawdown"].min() * 100
        mdd_df = pd.DataFrame({"Max Drawdown (%)": mdd_summary.round(2)})
        mdd_df.index = [STRAT_FULL_NAMES.get(k, k) for k in mdd_df.index]
        st.dataframe(mdd_df, use_container_width=True)

    st.markdown("#### ความแตกต่างระหว่างกลยุทธ์ เชื่อถือได้แค่ไหน (นัยสำคัญทางสถิติ)")
    n_comparisons = len(strategies) * (len(strategies) - 1) // 2
    bonferroni_alpha = 0.05 / n_comparisons if n_comparisons else 0.05
    sig_rows = []
    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            s1, s2 = strategies[i], strategies[j]
            diff = (wide[s1] - wide[s2]).dropna()
            t_stat, p_val = stats.ttest_rel(wide[s1], wide[s2])
            p_boot = block_bootstrap_pvalue(diff.values, block_size=4, n_boot=2000)
            sig_rows.append({
                "เปรียบเทียบ": f"{STRAT_FULL_NAMES.get(s1, s1)} vs {STRAT_FULL_NAMES.get(s2, s2)}",
                "p-value (t-test)": round(p_val, 3),
                "p-value (block bootstrap)": round(p_boot, 3) if not np.isnan(p_boot) else "N/A",
                "ผ่านเกณฑ์ Bonferroni?": "ผ่าน" if p_val < bonferroni_alpha else "ไม่ผ่าน",
                "สรุป (p<0.05)": "ต่างกันจริง" if p_val < 0.05 else "ยังสรุปไม่ได้ชัดเจน",
            })
    st.dataframe(pd.DataFrame(sig_rows), use_container_width=True, hide_index=True)
    st.caption(
        f"เทียบทั้งหมด {n_comparisons} คู่พร้อมกัน จึงใช้เกณฑ์ Bonferroni ที่เข้มกว่าปกติ (α ≈ {bonferroni_alpha:.4f}) "
        "กันไม่ให้สุ่มเจอผลบวกปลอมจากการเทียบหลายคู่พร้อมกัน ส่วน p-value (block bootstrap) ไม่ยึดสมมติฐานว่าแต่ละรอบเป็นอิสระจากกัน "
        "(เพราะ train ของรอบที่ติดกันทับซ้อนกันถึง 75%) ถ้าทั้งสองวิธีให้ข้อสรุปตรงกัน แปลว่าผลลัพธ์ทนทานต่อข้อสมมตินี้จริง"
    )

    st.markdown("#### เมทริกซ์ความสัมพันธ์ของสินทรัพย์ในพอร์ต (Correlation Heatmap)")
    corr_matrix = all_returns_full.corr()
    fig_corr, ax_corr = plt.subplots(figsize=(7, 4.5), facecolor='#0f172a')
    ax_corr.set_facecolor('#1e293b')
    cax = ax_corr.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    fig_corr.colorbar(cax)
    ax_corr.set_xticks(range(len(corr_matrix.columns)))
    ax_corr.set_yticks(range(len(corr_matrix.columns)))
    ax_corr.set_xticklabels(corr_matrix.columns, rotation=45, color='white', fontsize=8)
    ax_corr.set_yticklabels(corr_matrix.columns, color='white', fontsize=8)
    plt.title("Asset Return Correlation", color='white', fontsize=11, pad=15)
    st.pyplot(fig_corr)

# TAB 4: แชร์ผลลัพธ์
with tab4:
    st.subheader("📣 แชร์สรุปผลลัพธ์พอร์ตของคุณ")
    card_buf = create_share_card(best_key, best_return, best_sharpe, best_mdd, selected, capital, curr_name)
    st.image(card_buf, width=600)
    st.download_button(
        label="📥 ดาวน์โหลดรูปการ์ดสรุปผล (PNG)",
        data=card_buf,
        file_name="quantlab_summary.png",
        mime="image/png"
    )

# TAB 5: เกี่ยวกับข้อมูล
with tab5:
    st.subheader("ℹ️ ข้อมูลระบบและข้อจำกัดการทดสอบ")
    st.markdown("""
    - **Walk-Forward Validation**: จำลองการลงทุนเสมือนจริงโดยตัดข้อมูลออกเป็นช่วงๆ (Train และ Test) ป้องกันการรู้ล่วงหน้า (Look-ahead bias)
    - **Rebalance Cost**: คิดค่าธรรมเนียมซื้อขายเฉพาะสัดส่วนที่มีการเปลี่ยนแปลงจริง (Turnover Rate) เพื่อความสมจริง
    - **Currency Auto-detect**: ระบบตรวจสอบสกุลเงินอัตโนมัติจาก Suffix ของ Ticker หุ้น (.BK = THB, .T = JPY, .HK = HKD, ไม่มี Suffix = USD)
    """)

st.divider()
st.caption("QuantLab Analytics Engine — เครื่องมือจำลองพอร์ตการลงทุนเชิงปริมาณเพื่อการเรียนรู้และวิจัยทางสถิติ")
st.caption(
    "⚠️ ผลลัพธ์ทั้งหมดคำนวณจากข้อมูลราคาหุ้นในอดีต (backtest) เท่านั้น ไม่ใช่การรับประกันผลตอบแทนในอนาคต "
    "ยังไม่รวมภาษี เครื่องมือนี้จัดทำเพื่อการศึกษาและวิจัยทางสถิติ ไม่ใช่คำแนะนำการลงทุน และไม่ใช่บริการที่ได้รับอนุญาตจาก ก.ล.ต.\n\n"
    "ข้อสมมติที่ควรรู้: (1) กลยุทธ์ Market-cap ใช้จำนวนหุ้นที่ออกจำหน่ายปัจจุบัน คูณราคาย้อนหลัง เป็นค่าประมาณมูลค่าตลาดในอดีต "
    "(หุ้นที่หาข้อมูลไม่ได้จะใช้ค่ามัธยฐานของหุ้นอื่นในพอร์ตแทน) (2) แต่ละรอบ walk-forward ใช้ข้อมูล train ที่ทับซ้อนกันบางส่วน "
    "จึงไม่เป็นอิสระจากกันทั้งหมด ผลการทดสอบนัยสำคัญทางสถิติจึงควรตีความอย่างระมัดระวัง"
    )
