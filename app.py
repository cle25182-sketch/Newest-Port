import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
from datetime import datetime, timedelta
import streamlit as st

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="QuantLab — ระบบจำลองและจัดพอร์ตหุ้นเชิงปริมาณ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

RF_RATE = 0.0
MAX_TICKERS_ALLOWED = 10  # กลุ่ม D: จำกัดจำนวนหุ้นสูงสุดเพื่อป้องกันเซิร์ฟเวอร์ค้าง

# Custom Styling
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

# ==============================================================================
# 🛡️️ กลุ่ม A: ระบบกั้นหน้าจอข้อตกลงและเงื่อนไข (Disclaimer Gate)
# ==============================================================================
if "terms_accepted" not in st.session_state:
    st.session_state.terms_accepted = False

if not st.session_state.terms_accepted:
    st.title("📈 QuantLab — Portfolio Optimization Dashboard")
    st.markdown("### ⚠️ ข้อตกลง เงื่อนไขการใช้งาน และคำเตือนเรื่องความเสี่ยง")
    
    st.info("""
    **โปรดอ่านรายละเอียดก่อนเข้าใช้งาน:**
    
    1. **ไม่ใช่คำแนะนำการลงทุน (No Investment Advice):** เครื่องมือนี้จัดทำขึ้นเพื่อการจำลองทางสถิติและการเรียนรู้เชิงปริมาณเท่านั้น ไม่ใช่การให้คำแนะนำทางการเงิน การลงทุน หรือการชี้ชวนซื้อขายหลักทรัพย์ใดๆ
    2. **ผลงานในอดีตไม่ได้การันตีอนาคต:** ผลการทดสอบย้อนหลัง (Backtesting) เป็นการนำข้อมูลราคาในอดีตมาจำลองเท่านั้น ไม่สามารถยืนยันหรือรับประกันผลตอบแทนในอนาคตได้
    3. **ข้อจำกัดของแบบจำลอง:** การคำนวณตั้งอยู่บนสมมติฐานทางคณิตศาสตร์ ไม่ได้รวมปัจจัยเรื่องสภาพคล่อง อัตราภาษี เงินปันผล และสภาวะวิกฤตที่ไม่เคยเกิดขึ้นในอดีต
    4. **ความรับผิดชอบ:** ผู้พัฒนาแอปพลิเคชันจะไม่รับผิดชอบต่อความสูญเสียหรือความเสียหายใดๆ ที่เกิดจากการนำข้อมูลหรือผลลัพธ์จากเครื่องมือนี้ไปใช้ในการตัดสินใจลงทุนจริง
    """)
    
    st.markdown("---")
    agree = st.checkbox("ข้าพเจ้าได้อ่าน เข้าใจ และยอมรับว่าการใช้งานแอปพลิเคชันนี้เป็นไปเพื่อการศึกษาและจำลองข้อมูลเท่านั้น")
    
    if st.button("🚀 เข้าสู่ระบบวิเคราะห์พอร์ต", type="primary", disabled=not agree):
        st.session_state.terms_accepted = True
        st.rerun()
    
    st.stop()  # หยุดการทำงานไว้ที่นี่จนกว่าจะยอมรับเงื่อนไข

# ==============================================================================
# ⚙️ กลุ่ม D: Data Ingestion & Caching (ยืด TTL เป็น 12 ชม. + Error Handling)
# ==============================================================================

@st.cache_data(ttl=43200, show_spinner=False)
def load_price_data(tickers, years_back):
    """ดึงข้อมูลราคาหุ้นย้อนหลัง พร้อมระบบจัดการ Error และเพิ่ม TTL แคช 12 ชั่วโมง"""
    import yfinance as yf
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=years_back * 365)).strftime("%Y-%m-%d")
    
    try:
        raw = yf.download(list(tickers), start=start_date, end=end_date, progress=False, auto_adjust=True)["Close"]
        if raw.empty:
            return None, [], "ไม่พบข้อมูลราคาหุ้นสำหรับรหัสที่ระบุ"
        if isinstance(raw, pd.Series):
            raw = raw.to_frame(tickers[0])
        raw = raw.ffill()
        valid = [t for t in tickers if t in raw.columns and raw[t].notna().sum() >= 30]
        return raw[valid].dropna(), valid, None
    except Exception as e:
        return None, [], f"ไม่สามารถเชื่อมต่อกับ Yahoo Finance ได้ในขณะนี้ ({str(e)})"


@st.cache_data(ttl=43200, show_spinner=False)
def load_benchmark(years_back):
    """ดึงดัชนี SET Index จริงเป็น Benchmark ภายนอก"""
    import yfinance as yf
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=years_back * 365)).strftime("%Y-%m-%d")
    try:
        raw = yf.download("^SET.BK", start=start_date, end=end_date, progress=False, auto_adjust=True)["Close"]
        if isinstance(raw, pd.DataFrame):
            raw = raw.iloc[:, 0]
        return raw.dropna()
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=43200, show_spinner=False)
def load_market_caps(tickers):
    import yfinance as yf
    caps = {}
    for t in tickers:
        try:
            caps[t] = yf.Ticker(t).info.get("sharesOutstanding")
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
    """Ledoit-Wolf Covariance Shrinkage"""
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


def get_weights(train_ret, train_last_price, shares_arr, n, use_marketcap):
    mu, cov = train_ret.mean(), train_ret.cov()
    w_equal = np.array([1 / n] * n)

    def neg_sharpe(w, mu_arr, cov_arr):
        ret = np.sum(mu_arr * w) * 252 - RF_RATE
        vol = np.sqrt(max(w @ cov_arr @ w, 0)) * np.sqrt(252)
        return -ret / vol if vol > 1e-10 else 1e6

    bounds = tuple((0, 1) for _ in range(n))
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    opt = minimize(neg_sharpe, w_equal, args=(mu.values, cov.values),
                   method="SLSQP", bounds=bounds, constraints=constraints)
    w_markowitz = opt.x if opt.success else w_equal

    shrunk_cov, delta = ledoit_wolf_shrinkage(train_ret.values)
    opt_shrink = minimize(neg_sharpe, w_equal, args=(mu.values, shrunk_cov),
                           method="SLSQP", bounds=bounds, constraints=constraints)
    w_shrink = opt_shrink.x if opt_shrink.success else w_equal

    weights = {
        "A: Equal-weight (สัดส่วนเท่ากัน)": w_equal,
        "B: Markowitz (ปรับตาม Sharpe Max)": w_markowitz,
        "D: Markowitz-Shrinkage (ลด Noise)": w_shrink,
    }
    if use_marketcap:
        mcap = train_last_price * shares_arr
        weights["C: Market-cap (ตามมูลค่าบริษัท)"] = mcap / mcap.sum()
    return weights, (opt.success and opt_shrink.success), delta


def evaluate(w, test_ret):
    port_ret = test_ret.values @ w
    cum = np.cumprod(1 + port_ret)
    ann_ret = port_ret.mean() * 252
    ann_vol = port_ret.std() * np.sqrt(252)
    sharpe = (ann_ret - RF_RATE) / ann_vol if ann_vol > 1e-10 else np.nan
    running_max = np.maximum.accumulate(cum)
    max_dd = ((cum - running_max) / running_max).min()
    return cum[-1] - 1, ann_ret, ann_vol, sharpe, max_dd


def run_walk_forward(data, shares_arr, use_marketcap, train_window, test_window):
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
        weights, ok, delta = get_weights(train_ret, train_last_price, shares_arr, n, use_marketcap)
        failed += 0 if ok else 1
        deltas.append(delta)
        last_weights = weights
        for name, w in weights.items():
            weight_history.setdefault(name, []).append({"round": round_num, **{t: w[i] for i, t in enumerate(data.columns)}})
            total, ar, av, sh, mdd = evaluate(w, test_ret)
            records.append({
                "round": round_num,
                "strategy": name,
                "total_return": total,
                "ann_return": ar,
                "ann_vol": av,
                "sharpe": sh,
                "max_drawdown": mdd
            })
            port_ret_series = pd.Series(test_ret.values @ w, index=test_ret.index)
            daily_returns.setdefault(name, []).append(port_ret_series)
        start += test_window
    daily_returns = {k: pd.concat(v) for k, v in daily_returns.items()}
    weight_history = {k: pd.DataFrame(v).set_index("round") for k, v in weight_history.items()}
    avg_delta = float(np.mean(deltas)) if deltas else 0.0
    return pd.DataFrame(records), round_num, failed, daily_returns, last_weights, weight_history, avg_delta


def cumulative_growth(daily_returns, capital, cost_pct, test_window):
    curves = {}
    for name, ret_series in daily_returns.items():
        equity, values = capital, []
        for i, r in enumerate(ret_series.values):
            if i > 0 and i % test_window == 0:
                equity *= (1 - cost_pct)
            equity *= (1 + r)
            values.append(equity)
        curves[name] = pd.Series(values, index=ret_series.index)
    return curves


def block_bootstrap_pvalue(diff_values, block_size=4, n_boot=2000, seed=0):
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
    p_value = np.mean(np.abs(boot_centered) >= abs(observed_mean))
    return p_value

# ==============================================================================
# 📣 กลุ่ม C: ฟังก์ชั่นสร้าง Social Share Card (รูปสรุปผลลัพธ์)
# ==============================================================================
def create_share_card(best_strategy, annual_ret, sharpe, max_dd, tickers, capital):
    """สร้างภาพกราฟิกสวยงามสรุปผลลัพธ์เพื่อนำไปแชร์ต่อบน Social Media"""
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    ax.axis('off')
    
    # Title
    ax.text(0.05, 0.88, "QUANTLAB PORTFOLIO SUMMARY", fontsize=16, fontweight='bold', color='#38bdf8')
    ax.text(0.05, 0.80, f"การจำลองจัดพอร์ตด้วยหุ้น: {', '.join(tickers)}", fontsize=10, color='#94a3b8')
    
    # Highlight Strategy
    ax.text(0.05, 0.65, "กลยุทธ์ชนะเลิศ (Best Strategy):", fontsize=10, color='#cbd5e1')
    ax.text(0.05, 0.55, best_strategy.split("(")[0], fontsize=18, fontweight='bold', color='#4ade80')
    
    # Metric Boxes
    # Box 1: Return
    ax.text(0.05, 0.35, "ผลตอบแทนต่อปี", fontsize=9, color='#94a3b8')
    ax.text(0.05, 0.23, f"{annual_ret*100:+.1f}%", fontsize=16, fontweight='bold', color='#white')
    
    # Box 2: Sharpe
    ax.text(0.38, 0.35, "Sharpe Ratio", fontsize=9, color='#94a3b8')
    ax.text(0.38, 0.23, f"{sharpe:.2f}", fontsize=16, fontweight='bold', color='#facc15')
    
    # Box 3: Max DD
    ax.text(0.70, 0.35, "ขาดทุนสะสมสูงสุด", fontsize=9, color='#94a3b8')
    ax.text(0.70, 0.23, f"{max_dd*100:.1f}%", fontsize=16, fontweight='bold', color='#f87171')
    
    # Footer
    ax.text(0.05, 0.08, f"เงินเริ่มต้น: {capital:,.0f} THB | สร้างโดย QuantLab Analytics Dashboard", fontsize=8, color='#64748b')
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf

# ==============================================================================
# UI Interface Section
# ==============================================================================

st.title("📈 QuantLab — ระบบจำลองจัดพอร์ตหุ้นเชิงปริมาณ")
st.caption("ระบบจำลองพอร์ตการลงทุนแบบ Walk-Forward Validation ด้วยอัลกอริทึมทางคณิตศาสตร์และการเงิน")

# 🎓 กลุ่ม B: Onboarding Guide
with st.expander("👋 คำแนะนำเริ่มต้นใช้งานแบบรวดเร็ว (3 ขั้นตอนง่ายๆ)"):
    st.markdown("""
    1. **เลือกหุ้นที่สนใจ** ในเมนูทางซ้าย (เลือกได้สูงสุด 10 ตัว) หรือกดปุ่มตัวอย่างหุ้นไทย / หุ้นสหรัฐฯ
    2. **ใส่เงินลงทุนเริ่มต้น** และตั้งค่าระยะเวลาทดสอบย้อนหลัง
    3. **กดปุ่ม '🚀 เริ่มวิเคราะห์พอร์ต'** เพื่อดูว่ากลยุทธ์ไหนให้ผลลัพธ์คุ้มค่าความเสี่ยงมากที่สุด!
    """)

# 🎓 กลุ่ม B: โหมดอธิบายภาษาคน (ELI5 Explanations)
simple_mode = st.toggle("💡 เปิดโหมดอธิบายภาษาพูด (สำหรับผู้เริ่มต้นที่ไม่มีพื้นฐานการเงิน)")

if simple_mode:
    st.markdown("""
    <div class="concept-box">
    <b>💡 คู่มือความหมายฉบับเข้าใจง่าย:</b><br>
    • <b>Sharpe Ratio (คะแนนความคุ้มค่า):</b> ยิ่งสูง ยิ่งดี! เหมือนการซื้อของที่ได้ของคุณภาพดีเยี่ยมในราคาคุ้มเงิน<br>
    • <b>Volatility (ความแกว่งตัว):</b> เหมือนความหวาดเสียวของการนั่งรถไฟเหาะ ยิ่งเปอร์เซ็นต์สูง แปลว่าราคาขึ้นลงน่ากลัว<br>
    • <b>Max Drawdown (สถิติเจ็บหนักสุด):</b> บอกว่าในอดีต เงินเคยลดลงจากจุดสูงสุดลึกลงไปกี่ % ก่อนจะฟื้นกลับมา<br>
    • <b>p-value (โอกาสฟลุ๊ก):</b> ถ้าน้อยกว่า 0.05 แปลว่าผลตอบแทนที่ดีนั้นเกิดจาก "ระบบเก่งจริง" ไม่ใช่แค่ "โชคดี"
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Sidebar Configuration
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ ตั้งค่าการวิเคราะห์")
    capital = st.number_input("เงินลงทุนเริ่มต้น (บาท)", min_value=1000, value=100000, step=5000)

    if "ticker_text" not in st.session_state:
        st.session_state.ticker_text = "PTT.BK, CPALL.BK, AOT.BK, KBANK.BK, ADVANC.BK"

    st.caption("ชุดตัวอย่างด่วน:")
    preset_col1, preset_col2 = st.columns(2)
    with preset_col1:
        if st.button("🇹🇭 หุ้นไทย (Top 5)", use_container_width=True):
            st.session_state.ticker_text = "PTT.BK, CPALL.BK, AOT.BK, KBANK.BK, ADVANC.BK"
    with preset_col2:
        if st.button("🇺🇸 หุ้นสหรัฐฯ (Big Tech)", use_container_width=True):
            st.session_state.ticker_text = "AAPL, MSFT, GOOGL, AMZN, NVDA"

    ticker_input = st.text_input(
        "พิมพ์รหัสหุ้น (คั่นด้วยจุลภาค ,)",
        key="ticker_text",
        help="ใส่รหัสหุ้น Yahoo Finance เช่น PTT.BK, AAPL"
    )
    selected_raw = list(dict.fromkeys([t.strip().upper() for t in ticker_input.split(",") if t.strip()]))

    # ⚙️ กลุ่ม D: จำกัดจำนวนหุ้นสูงสุด 10 ตัว
    if len(selected_raw) > MAX_TICKERS_ALLOWED:
        st.warning(f"⚠️ เพื่อความรวดเร็วและป้องกันเซิร์ฟเวอร์ทำงานหนัก ระบบจำกัดสูงสุดที่ {MAX_TICKERS_ALLOWED} หุ้น (เลือก {MAX_TICKERS_ALLOWED} ตัวแรกให้)")
        selected = selected_raw[:MAX_TICKERS_ALLOWED]
    else:
        selected = selected_raw

    with st.expander("ตั้งค่าการทดสอบขั้นสูง"):
        train_window = st.slider("ช่วง Train ข้อมูล (วัน)", 126, 378, 252, step=21)
        test_window = st.slider("ช่วง Test ต่อรอบ (วัน)", 21, 126, 63, step=21)
        years_back = st.slider("ข้อมูลย้อนหลัง (ปี)", 3, 10, 5)
        cost_pct = st.slider("ค่าธรรมเนียมซื้อขายปรับพอร์ต (%)", 0.0, 1.0, 0.1, step=0.05) / 100
        stress_option = st.selectbox(
            "จำลองวิกฤตเฉพาะช่วง",
            ["ไม่ระบุ", "COVID-19 (ก.พ.–เม.ย. 2020)", "เงินเฟ้อ/ดอกเบี้ยขาขึ้น (2022)"]
        )

    run = st.button("🚀 เริ่มวิเคราะห์พอร์ต", type="primary", use_container_width=True)

# ------------------------------------------------------------------------------
# Execution & Results Validation
# ------------------------------------------------------------------------------
if not run:
    st.info("👈 ปรับแต่งตัวเลือกทางซ้ายมือ แล้วกด **เริ่มวิเคราะห์พอร์ต** เพื่อประมวลผล")
    st.stop()

if len(selected) < 2:
    st.error("กรุณาใส่รหัสหุ้นอย่างน้อย 2 ตัวขึ้นไปเพื่อจัดพอร์ตกระจายความเสี่ยง")
    st.stop()

# 🛡️ กลุ่ม A: การรับมือ Error ของ Yahoo Finance อย่างเป็นมิตร
with st.spinner("กำลังดึงราคาหุ้นและประมวลผลทางสถิติ..."):
    data, valid_tickers, error_msg = load_price_data(tuple(selected), years_back)
    
    if error_msg:
        st.error(f"⚠️ การดึงข้อมูลล้มเหลว: {error_msg}")
        st.info("💡 **ข้อแนะนำในการแก้ไข:**\n- ลองเว้นระยะเวลา 1-2 นาทีแล้วกดใหม่อีกครั้ง (กรณีติด Rate Limit)\n- ตรวจสอบว่ารหัสหุ้นถูกต้อง เช่น หุ้นไทยต้องลงท้ายด้วย `.BK` (เช่น PTT.BK)")
        st.stop()

selected = valid_tickers
if len(selected) < 2:
    st.error("เหลือหุ้นที่ข้อมูลสมบูรณ์น้อยกว่า 2 ตัว กรุณาเปลี่ยนรหัสหุ้น")
    st.stop()

caps_raw = load_market_caps(tuple(selected))
clean_caps = {t: _clean_shares(caps_raw.get(t)) for t in selected}
use_marketcap = all(clean_caps[t] is not None for t in selected)
shares_arr = np.array([clean_caps[t] or 0 for t in selected])

df, n_folds, n_failed, daily_returns, last_weights, weight_history, avg_delta = run_walk_forward(
    data, shares_arr, use_marketcap, train_window, test_window
)

if df.empty:
    st.error("ข้อมูลย้อนหลังมีไม่เพียงพอต่อการแบ่งรอบทดสอบ ลองลดช่วงวัน Train/Test ในเมนูขั้นสูง")
    st.stop()

strategies = list(df["strategy"].unique())
summary = df.groupby("strategy")[["ann_return", "ann_vol", "sharpe", "max_drawdown"]].mean().reindex(strategies)
best_strategy = summary["sharpe"].idxmax()
best_return = summary.loc[best_strategy, "ann_return"]
best_vol = summary.loc[best_strategy, "ann_vol"]
best_sharpe = summary.loc[best_strategy, "sharpe"]
best_mdd = summary.loc[best_strategy, "max_drawdown"]

# ------------------------------------------------------------------------------
# Output Tabs
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 สรุปผลพอร์ต", 
    "📈 การเติบโตของเงินทุน", 
    "🔬 วิเคราะห์สถิติเชิงลึก", 
    "📣 แชร์ผลลัพธ์",
    "ℹ️ เกี่ยวกับข้อมูล & ข้อจำกัด"
])

# ================= TAB 1: สรุปผล =================
with tab1:
    st.subheader(f"🏆 กลยุทธ์ที่ทำผลงานได้ดีที่สุด: {best_strategy}")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("ผลตอบแทนเฉลี่ย/ปี", f"{best_return * 100:.1f}%")
    with col_m2:
        st.metric("ความผันผวน/ปี", f"{best_vol * 100:.1f}%")
    with col_m3:
        st.metric("Sharpe Ratio", f"{best_sharpe:.2f}")
    with col_m4:
        st.metric("Max Drawdown", f"{best_mdd * 100:.1f}%")

    st.markdown("#### สัดส่วนจัดสรรเงินลงทุนล่าสุด (Action Plan)")
    winner_weights = last_weights[best_strategy]
    action_plan = []
    for ticker, weight in zip(selected, winner_weights):
        action_plan.append({
            "รหัสหุ้น": ticker,
            "สัดส่วน (%)": f"{weight * 100:.2f}%",
            "จำนวนเงินจัดซื้อ (บาท)": f"{capital * weight:,.2f}",
        })
    st.dataframe(pd.DataFrame(action_plan), use_container_width=True, hide_index=True)

    st.markdown("#### เปรียบเทียบผลลัพธ์ทุกกลยุทธ์ (ค่าเฉลี่ย)")
    display_summary = summary.copy()
    display_summary["ann_return"] = (display_summary["ann_return"] * 100).round(1).astype(str) + "%"
    display_summary["ann_vol"] = (display_summary["ann_vol"] * 100).round(1).astype(str) + "%"
    display_summary["max_drawdown"] = (display_summary["max_drawdown"] * 100).round(1).astype(str) + "%"
    display_summary["sharpe"] = display_summary["sharpe"].round(2)
    display_summary.columns = ["ผลตอบแทน/ปี", "ความผันผวน/ปี", "Sharpe Ratio", "Max Drawdown"]
    st.dataframe(display_summary, use_container_width=True)

# ================= TAB 2: การเติบโตของเงินทุน =================
with tab2:
    st.subheader("เส้นทางการเติบโตของเงินทุน (คิดทบต้นจริง + ค่าธรรมเนียม)")
    curves = cumulative_growth(daily_returns, capital, cost_pct, test_window)
    bench_raw = load_benchmark(years_back)

    fig_cum, ax_cum = plt.subplots(figsize=(10, 4.5))
    for name, curve in curves.items():
        ax_cum.plot(curve.index, curve.values, label=name.split(" ")[0], linewidth=1.6)

    # Benchmark Alignment
    if not bench_raw.empty:
        combined_index = next(iter(curves.values())).index
        bench_aligned = bench_raw.reindex(bench_raw.index.union(combined_index)).ffill().reindex(combined_index)
        if bench_aligned.notna().sum() > 10:
            bench_ret = bench_aligned.pct_change().fillna(0)
            bench_curve = capital * (1 + bench_ret).cumprod()
            ax_cum.plot(bench_curve.index, bench_curve.values, label="SET Index (Buy & Hold)", linewidth=1.8, linestyle="--", color="black")

    # 📣 กลุ่ม C: ใส่เหตุการณ์สำคัญทางเศรษฐกิจบนกราฟ (Timeline Annotations)
    ax_cum.axhline(capital, color="gray", linewidth=0.7, linestyle=":")
    
    # Check if dates exist in series to annotate
    curve_dates = next(iter(curves.values())).index
    if any(d.year == 2020 for d in curve_dates):
        ax_cum.axvspan(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-30"), color="red", alpha=0.1)
        ax_cum.text(pd.Timestamp("2020-02-15"), capital*0.9, "COVID-19 Crash", color="red", fontsize=8)
    if any(d.year == 2022 for d in curve_dates):
        ax_cum.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"), color="orange", alpha=0.1)
        ax_cum.text(pd.Timestamp("2022-03-01"), capital*1.1, "Global Rate Hikes", color="orange", fontsize=8)

    ax_cum.set_xlabel("Date")
    ax_cum.set_ylabel("Portfolio Value (THB)")
    ax_cum.set_title("Cumulative Equity Curve")
    ax_cum.legend(loc="upper left", fontsize=8)
    st.pyplot(fig_cum)

# ================= TAB 3: วิเคราะห์เชิงลึก =================
with tab3:
    st.subheader("การทดสอบนัยสำคัญทางสถิติ (Statistical Validation)")
    
    wide = df.pivot(index="round", columns="strategy", values="sharpe")
    rows = []
    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            s1, s2 = strategies[i], strategies[j]
            diff = (wide[s1] - wide[s2]).dropna()
            _, p_val = stats.ttest_rel(wide[s1], wide[s2])
            p_boot = block_bootstrap_pvalue(diff.values, block_size=4, n_boot=1000)
            rows.append({
                "คู่เปรียบเทียบ": f"{s1.split(' ')[0]} vs {s2.split(' ')[0]}",
                "p-value (Paired t-test)": round(p_val, 4),
                "p-value (Block Bootstrap)": round(p_boot, 4) if not np.isnan(p_boot) else "N/A",
                "สรุปนัยสำคัญ (p < 0.05)": "ต่างกันอย่างมีนัยสำคัญ" if p_val < 0.05 else "ยังไม่พบความต่างชัดเจน"
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ================= TAB 4: แชร์ผลลัพธ์ (Social Sharing) =================
with tab4:
    st.subheader("📣 แชร์สรุปผลลัพธ์พอร์ตของคุณ")
    st.write("ดาวน์โหลดภาพการ์ดสรุปผลการทดสอบ เพื่อนำไปแบ่งปันหรือโพสต์ต่อบนโซเชียลมีเดีย")
    
    card_buf = create_share_card(best_strategy, best_return, best_sharpe, best_mdd, selected, capital)
    st.image(card_buf, caption="ตัวอย่าง Social Share Card", use_container_width=False, width=600)
    
    st.download_button(
        label="📥 ดาวน์โหลดรูปการ์ดสรุปผล (PNG)",
        data=card_buf,
        file_name="quantlab_portfolio_summary.png",
        mime="image/png"
    )

# ================= TAB 5: เกี่ยวกับข้อมูล & ข้อจำกัด =================
with tab5:
    st.subheader("ℹ️ แหล่งที่มาของข้อมูล และข้อจำกัดของระบบ")
    st.markdown("""
    #### 1. แหล่งข้อมูลราคา (Data Ingestion)
    • ข้อมูลราคาหุ้นดึงผ่าน **Yahoo Finance API** โดยใช้ราคาปิดปรับปรุง (Adjusted Close Price) เพื่อสะท้อนผลปันผลและการแตกหุ้น  
    • ระบบแคชข้อมูลไว้เป็นเวลา 12 ชั่วโมง เพื่อความรวดเร็วและหลีกเลี่ยงการติด Rate Limit  

    #### 2. สมมติฐานทางคณิตศาสตร์และการเงิน
    • **Risk-Free Rate ($R_f$):** กำหนดไว้ที่ $0.0\%$ เพื่อความเรียบง่ายในการเปรียบเทียบผลตอบแทนส่วนเกิน  
    • **Rebalance Cost:** มีการหักค่าธรรมเนียมการซื้อขายตามอัตราที่ผู้ใช้กำหนดทุกครั้งที่มีการปรับพอร์ตในต้นรอบใหม่  
    • **Market Capitalization:** กลยุทธ์ C ใช้จำนวนหุ้นชำระแล้ว (Shares Outstanding) ปัจจุบัน เป็นตัวคูณประมาณการมูลค่าตลาดในอดีต  

    #### 3. คำเตือนความเสี่ยง (Risk Disclaimer)
    • ผลการจำลองนี้ไม่รวมปัจจัยเรื่อง **Slippage** (ความต่างของราคาที่ส่งคำสั่งกับราคาที่จับคู่ได้จริง)  
    • ไม่รวมภาระภาษีเงินปันผลหรือภาษีลาภลอย (Capital Gains Tax)  
    • การทดสอบย้อนหลังเป็นเพียงเครื่องมือช่วยศึกษารูปแบบสถิติในอดีต **ไม่สามารถใช้เป็นสิ่งยืนยันผลตอบแทนในอนาคตได้**
    """)

st.divider()
st.caption("QuantLab Analytics Engine — เครื่องมือจำลองพอร์ตการลงทุนเชิงปริมาณเพื่อการเรียนรู้และวิจัยทางสถิติ")
```
