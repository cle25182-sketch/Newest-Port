import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
from datetime import datetime, timedelta
import streamlit as st

# ==============================================================================
# Page Configuration & Global Settings
# ==============================================================================
st.set_page_config(
    page_title="QuantLab — ระบบจำลองและจัดพอร์ตหุ้นเชิงปริมาณ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

RF_RATE = 0.0
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

# ==============================================================================
# 🛡️ Disclaimer Gate (กั้นหน้าจอคำเตือนเรื่องความเสี่ยง)
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
    
    st.stop()

# ==============================================================================
# Data Ingestion & Shrinkage Analytics Engine
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


def get_weights(train_ret, train_last_price, shares_arr, n, use_marketcap):
    """คำนวณน้ำหนักพอร์ตสำหรับทั้ง 4 กลยุทธ์"""
    mu, cov = train_ret.mean(), train_ret.cov()
    w_equal = np.array([1 / n] * n)

    def neg_sharpe(w, mu_arr, cov_arr):
        ret = np.sum(mu_arr * w) * 252 - RF_RATE
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
        "A: Equal-weight (สัดส่วนเท่ากัน)": w_equal,
        "B: Markowitz (สมการคณิตศาสตร์)": w_markowitz,
        "D: Markowitz-Shrinkage (ลดสัญญาณรบกวน)": w_shrink,
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
    return pd.DataFrame(records), round_num, failed, daily_returns, last_weights, weight_history, avg_delta, all_returns


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
# 🎨 Visualization Generator Functions (รูปการ์ดและกราฟวิเคราะห์เชิงลึก)
# ==============================================================================
def create_share_card(best_strategy, annual_ret, sharpe, max_dd, tickers, capital):
    """สร้างภาพการ์ดสรุปผลลัพธ์ด้วยตัวอักษรภาษาอังกฤษเพื่อป้องกันฟอนต์สี่เหลี่ยม (Missing Glyph) บน Cloud"""
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    ax.axis('off')

    # Header
    ax.text(0.05, 0.88, "QUANTLAB PORTFOLIO SUMMARY", fontsize=15, fontweight='bold', color='#38bdf8')
    ticker_str = ", ".join(tickers[:6]) + ("..." if len(tickers) > 6 else "")
    ax.text(0.05, 0.80, f"Assets: {ticker_str}", fontsize=9, color='#94a3b8')

    # Strategy Title
    ax.text(0.05, 0.66, "BEST STRATEGY", fontsize=9, fontweight='bold', color='#cbd5e1')
    strat_clean = best_strategy.split("(")[0].strip()
    ax.text(0.05, 0.54, strat_clean, fontsize=16, fontweight='bold', color='#4ade80')

    # Metrics
    ax.text(0.05, 0.35, "ANNUAL RETURN", fontsize=8, color='#94a3b8')
    ax.text(0.05, 0.22, f"{annual_ret*100:+.1f}%", fontsize=15, fontweight='bold', color='white')

    ax.text(0.38, 0.35, "SHARPE RATIO", fontsize=8, color='#94a3b8')
    ax.text(0.38, 0.22, f"{sharpe:.2f}", fontsize=15, fontweight='bold', color='#facc15')

    ax.text(0.70, 0.35, "MAX DRAWDOWN", fontsize=8, color='#94a3b8')
    ax.text(0.70, 0.22, f"{max_dd*100:.1f}%", fontsize=15, fontweight='bold', color='#f87171')

    # Footer
    ax.text(0.05, 0.08, f"Initial Capital: {capital:,.0f} THB  |  Generated by QuantLab Analytics Engine", fontsize=8, color='#64748b')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def win_tally_fig(df, strategies):
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    
    piv = df.pivot(index='round', columns='strategy', values='sharpe')
    winners = piv.idxmax(axis=1)
    
    tally = pd.DataFrame(0, index=piv.index, columns=strategies)
    for r in piv.index:
        w = winners.loc[r]
        tally.loc[r, w] = 1
    
    cum_tally = tally.cumsum()
    colors = ['#38bdf8', '#facc15', '#4ade80', '#c084fc']
    for i, strat in enumerate(strategies):
        short_name = strat.split(" ")[0]
        ax.plot(cum_tally.index, cum_tally[strat], label=short_name, color=colors[i % len(colors)], linewidth=2)
        
    ax.set_title("Cumulative Wins (Highest Sharpe Per Round)", color='white', fontsize=11, fontweight='bold')
    ax.set_xlabel("Walk-Forward Round", color='#94a3b8')
    ax.set_ylabel("Wins Count", color='#94a3b8')
    ax.tick_params(colors='#cbd5e1')
    ax.grid(True, linestyle='--', alpha=0.2)
    ax.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='white', fontsize=8)
    
    plt.tight_layout()
    return fig


def correlation_matrix_fig(all_returns):
    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    
    corr = all_returns.corr()
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax)
    
    tickers = corr.columns.tolist()
    ax.set_xticks(range(len(tickers)))
    ax.set_yticks(range(len(tickers)))
    ax.set_xticklabels(tickers, rotation=45, color='white', ha='left', fontsize=8)
    ax.set_yticklabels(tickers, color='white', fontsize=8)
    
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            val = corr.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='black' if abs(val) < 0.6 else 'white', fontsize=8)
            
    ax.set_title("Asset Return Correlation Matrix", color='white', fontsize=11, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig

def efficient_frontier_fig(all_returns, last_weights):
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    
    mu = all_returns.mean() * 252
    cov = all_returns.cov() * 252
    n_assets = len(mu)
    
    np.random.seed(42)
    n_portfolios = 2500
    p_rets, p_vols, p_sharpes = [], [], []
    
    for _ in range(n_portfolios):
        w = np.random.dirichlet(np.ones(n_assets))
        ret = np.sum(mu * w)
        vol = np.sqrt(w @ cov @ w)
        sharpe = ret / vol if vol > 1e-6 else 0
        p_rets.append(ret)
        p_vols.append(vol)
        p_sharpes.append(sharpe)
        
    sc = ax.scatter(np.array(p_vols)*100, np.array(p_rets)*100, c=p_sharpes, cmap='viridis', alpha=0.4, s=15)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Sharpe Ratio', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax, 'yticklabels'), color='white')
    
    markers = ['o', 's', '^', 'D']
    colors = ['#38bdf8', '#facc15', '#4ade80', '#c084fc']
    
    for i, (strat_name, w) in enumerate(last_weights.items()):
        strat_ret = np.sum(mu * w) * 100
        strat_vol = np.sqrt(w @ cov @ w) * 100
        short_name = strat_name.split(" ")[0]
        ax.scatter(strat_vol, strat_ret, color=colors[i % len(colors)], marker=markers[i % len(markers)],
                   s=120, edgecolors='white', linewidth=1.5, label=f"{short_name} (Latest)", zorder=5)
        
    ax.set_title("Simulated Efficient Frontier & Strategy Positions", color='white', fontsize=11, fontweight='bold')
    ax.set_xlabel("Annualized Volatility (%)", color='#94a3b8')
    ax.set_ylabel("Annualized Return (%)", color='#94a3b8')
    ax.tick_params(colors='#cbd5e1')
    ax.grid(True, linestyle='--', alpha=0.2)
    ax.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='white', fontsize=8, loc='upper left')
    
    plt.tight_layout()
    return fig


def weight_evolution_fig(weight_df, tickers, title):
    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    
    rounds = weight_df.index
    weights = [weight_df[t].values for t in tickers if t in weight_df.columns]
    valid_tickers = [t for t in tickers if t in weight_df.columns]
    
    if weights:
        ax.stackplot(rounds, weights, labels=valid_tickers, alpha=0.85)
        
    ax.set_title(f"Weight Evolution: {title}", color='white', fontsize=10, fontweight='bold')
    ax.set_xlabel("Round", color='#94a3b8', fontsize=8)
    ax.set_ylabel("Allocation Ratio", color='#94a3b8', fontsize=8)
    ax.tick_params(colors='#cbd5e1', labelsize=8)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', facecolor='#0f172a', edgecolor='#334155', labelcolor='white', fontsize=7, bbox_to_anchor=(1.35, 1))
    
    plt.tight_layout()
    return fig

# ==============================================================================
# UI Interface Section
# ==============================================================================
st.title("📈 QuantLab — ระบบจำลองจัดพอร์ตหุ้นเชิงปริมาณ")
st.caption("ระบบจำลองพอร์ตการลงทุนแบบ Walk-Forward Validation ด้วยอัลกอริทึมทางคณิตศาสตร์และการเงิน")

with st.expander("👋 คำแนะนำเริ่มต้นใช้งานแบบรวดเร็ว (3 ขั้นตอนง่ายๆ)"):
    st.markdown("""
    1. **เลือกหุ้นที่สนใจ** ในเมนูทางซ้าย (เลือกได้สูงสุด 10 ตัว) หรือกดปุ่มตัวอย่างหุ้นไทย / หุ้นสหรัฐฯ
    2. **ใส่เงินลงทุนเริ่มต้น** และตั้งค่าระยะเวลาทดสอบย้อนหลัง
    3. **กดปุ่ม '🚀 เริ่มวิเคราะห์พอร์ต'** เพื่อดูว่ากลยุทธ์ไหนให้ผลลัพธ์คุ้มค่าความเสี่ยงมากที่สุด!
    """)

simple_mode = st.toggle("💡 เปิดโหมดอธิบายภาษาพูด (สำหรับผู้เริ่มต้นที่ไม่มีพื้นฐานการเงิน)")

if simple_mode:
    st.markdown("""
    <div class="concept-box">
    <b>💡 คู่มือความหมายฉบับเข้าใจง่าย:</b><br>
    • <b>Sharpe Ratio (คะแนนความคุ้มค่า):</b> ยิ่งสูง ยิ่งดี! เหมือนการซื้อของที่ได้ของคุณภาพดีเยี่ยมในราคาคุ้มเงิน<br>
    • <b>Volatility (ความแกว่งตัว):</b> ยิ่งเปอร์เซ็นต์สูง แปลว่าราคาขึ้นลงน่ากลัวตามความเสี่ยง<br>
    • <b>Max Drawdown (สถิติเจ็บหนักสุด):</b> บอกว่าในอดีต เงินเคยลดลงจากจุดสูงสุดลึกลงไปกี่ % ก่อนจะฟื้นกลับมา<br>
    • <b>p-value (โอกาสฟลุ๊ก):</b> ถ้าน้อยกว่า 0.05 แปลว่าผลตอบแทนที่ดีนั้นเกิดจาก "ระบบเก่งจริง" ไม่ใช่แค่ "โชคดี"
    </div>
    """, unsafe_allow_html=True)

# Sidebar Configuration
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

    if len(selected_raw) > MAX_TICKERS_ALLOWED:
        st.warning(f"⚠️ เพื่อความรวดเร็ว ระบบจำกัดสูงสุดที่ {MAX_TICKERS_ALLOWED} หุ้น (เลือก {MAX_TICKERS_ALLOWED} ตัวแรกให้)")
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

if not run:
    st.info("👈 ปรับแต่งตัวเลือกทางซ้ายมือ แล้วกด **เริ่มวิเคราะห์พอร์ต** เพื่อประมวลผล")
    st.stop()

if len(selected) < 2:
    st.error("กรุณาใส่รหัสหุ้นอย่างน้อย 2 ตัวขึ้นไปเพื่อจัดพอร์ตกระจายความเสี่ยง")
    st.stop()

with st.spinner("กำลังดึงราคาหุ้นและประมวลผลทางสถิติ..."):
    data, valid_tickers, error_msg = load_price_data(tuple(selected), years_back)
    
    if error_msg:
        st.error(f"⚠️ การดึงข้อมูลล้มเหลว: {error_msg}")
        st.info("💡 **ข้อแนะนำในการแก้ไข:**\n- ลองเว้นระยะเวลา 1-2 นาทีแล้วกดใหม่อีกครั้ง\n- ตรวจสอบว่ารหัสหุ้นถูกต้อง เช่น หุ้นไทยต้องลงท้ายด้วย `.BK` (เช่น PTT.BK)")
        st.stop()

selected = valid_tickers
if len(selected) < 2:
    st.error("เหลือหุ้นที่ข้อมูลสมบูรณ์น้อยกว่า 2 ตัว กรุณาเปลี่ยนรหัสหุ้น")
    st.stop()

caps_raw = load_market_caps(tuple(selected))
clean_caps = {t: _clean_shares(caps_raw.get(t)) for t in selected}
use_marketcap = all(clean_caps[t] is not None for t in selected)
shares_arr = np.array([clean_caps[t] or 0 for t in selected])

df, n_folds, n_failed, daily_returns, last_weights, weight_history, avg_delta, all_returns_full = run_walk_forward(
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

# Calculate pairwise tests for best strategy significance
wide = df.pivot(index="round", columns="strategy", values="sharpe")
significantly_better_than_all = True
for strat in strategies:
    if strat != best_strategy:
        diff = wide[best_strategy] - wide[strat]
        _, p_val = stats.ttest_rel(wide[best_strategy], wide[strat])
        if p_val >= 0.05:
            significantly_better_than_all = False
            break

avg_corr = float(all_returns_full.corr().values[np.triu_indices_from(all_returns_full.corr().values, k=1)].mean())

# ==============================================================================
# Output Tabs Display
# ==============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 สรุปผลพอร์ต", 
    "📈 การเติบโตของเงินทุน", 
    "🔬 วิเคราะห์สถิติเชิงลึก", 
    "📣 แชร์ผลลัพธ์",
    "ℹ️ เกี่ยวกับข้อมูล & ข้อจำกัด"
])

# ================= TAB 1: สรุปผล =================
with tab1:
    st.subheader("สรุปผล: กลยุทธ์ที่ Sharpe ดีที่สุดในการทดสอบนี้")
    if significantly_better_than_all:
        st.success(
            f"🏆 **{best_strategy}** ให้ Sharpe ratio ดีที่สุด และแตกต่างจากกลยุทธ์อื่นทุกตัวอย่างมีนัยสำคัญทางสถิติ (p < 0.05) "
            f"ในการทดสอบ {n_folds} รอบนี้"
        )
    else:
        st.info(
            f"💡 **{best_strategy}** ให้ Sharpe ratio เฉลี่ยสูงสุดในการทดสอบนี้ แต่ยัง**ไม่ต่างจากกลยุทธ์อื่นอย่างมีนัยสำคัญทางสถิติ** "
            f"(ดูตาราง p-value ในแท็บ 'วิเคราะห์เชิงลึก') — ควรตีความว่า \"ยังแยกไม่ออกชัดเจนว่าวิธีไหนดีกว่าจริง\" มากกว่าฟันธงว่าตัวนี้ชนะ"
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

    st.markdown("#### ตัวอย่างการจัดสรรเงินตามกลยุทธ์ชนะเลิศ")
    winner_weights = last_weights[best_strategy]
    action_plan = []
    for ticker, weight in zip(selected, winner_weights):
        action_plan.append({
            "รหัสหุ้น": ticker,
            "สัดส่วน (%)": f"{weight * 100:.2f}%",
            "จำนวนเงิน (บาท)": f"{capital * weight:,.2f}",
        })
    st.dataframe(pd.DataFrame(action_plan), use_container_width=True, hide_index=True)

    st.markdown("#### ตารางเปรียบเทียบทุกกลยุทธ์ (ค่าเฉลี่ยตลอดทุกรอบ)")
    display_summary = summary.copy()
    display_summary["ann_return"] = (display_summary["ann_return"] * 100).round(1).astype(str) + "%"
    display_summary["ann_vol"] = (display_summary["ann_vol"] * 100).round(1).astype(str) + "%"
    display_summary["max_drawdown"] = (display_summary["max_drawdown"] * 100).round(1).astype(str) + "%"
    display_summary["sharpe"] = display_summary["sharpe"].round(2)
    display_summary.columns = ["ผลตอบแทน/ปี", "ความผันผวน/ปี", "Sharpe ratio", "Max Drawdown"]
    st.dataframe(display_summary, use_container_width=True)

    st.info(
        "**A: Equal-weight** — แบ่งเงินเท่ากันทุกตัว  \n"
        "**B: Markowitz** — คำนวณสัดส่วนด้วยสูตรคณิตศาสตร์ให้ Sharpe ratio สูงสุด  \n"
        "**C: Market-cap** — ถ่วงน้ำหนักตามมูลค่าบริษัท ใหญ่กว่าได้สัดส่วนมากกว่า  \n"
        "**D: Markowitz-Shrinkage** — เหมือน B แต่ลด estimation error ของข้อมูลด้วยเทคนิค Ledoit-Wolf shrinkage"
    )
    st.caption(f"ค่าสหสัมพันธ์เฉลี่ยระหว่างหุ้นที่เลือก: **{avg_corr:.2f}** (ยิ่งต่ำ ยิ่งกระจายความเสี่ยงได้ผลดี)")

# ================= TAB 2: การเติบโตของเงินทุน =================
with tab2:
    st.subheader("เงินลงทุนสะสม ถ้าลงทุนต่อเนื่องตลอดช่วงทดสอบ (คิดทบต้นจริง + ค่าธรรมเนียม)")
    curves = cumulative_growth(daily_returns, capital, cost_pct, test_window)
    bench_raw = load_benchmark(years_back)

    fig_cum, ax_cum = plt.subplots(figsize=(10, 4.5), facecolor='#0f172a')
    ax_cum.set_facecolor('#1e293b')
    colors = ['#38bdf8', '#facc15', '#4ade80', '#c084fc']
    
    for i, (name, curve) in enumerate(curves.items()):
        ax_cum.plot(curve.index, curve.values, label=name.split(" ")[0], color=colors[i % len(colors)], linewidth=1.6)

    bench_note = ""
    if not bench_raw.empty:
        combined_index = next(iter(curves.values())).index
        bench_aligned = bench_raw.reindex(bench_raw.index.union(combined_index)).ffill().reindex(combined_index)
        if bench_aligned.notna().sum() > 10:
            bench_ret = bench_aligned.pct_change().fillna(0)
            bench_curve = capital * (1 + bench_ret).cumprod()
            ax_cum.plot(bench_curve.index, bench_curve.values, label="SET Index (Buy & Hold)", linewidth=1.8,
                        linestyle="--", color="white")
        else:
            bench_note = "ข้อมูล SET Index ไม่พอสำหรับช่วงเวลานี้"
    else:
        bench_note = "ดึงข้อมูล SET Index (^SET.BK) ไม่สำเร็จ"

    ax_cum.axhline(capital, color="gray", linewidth=0.7, linestyle=":")
    ax_cum.set_xlabel("Date", color='#94a3b8')
    ax_cum.set_ylabel(f"Portfolio Value (THB)", color='#94a3b8')
    ax_cum.set_title(f"Cumulative Growth -- Includes {cost_pct*100:.2f}% Cost Per Rebalance", color='white', fontweight='bold')
    ax_cum.tick_params(colors='#cbd5e1')
    ax_cum.grid(True, linestyle='--', alpha=0.2)
    ax_cum.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='white', loc="upper left", fontsize=9)
    st.pyplot(fig_cum)
    
    if bench_note:
        st.caption(f"⚠️ {bench_note}")

    if stress_option != "ไม่ระบุ":
        st.subheader(f"ทดสอบเฉพาะช่วงวิกฤต: {stress_option}")
        stress_ranges = {
            "COVID-19 (ก.พ.–เม.ย. 2020)": ("2020-02-01", "2020-04-30"),
            "เงินเฟ้อ/ดอกเบี้ยขาขึ้น (2022)": ("2022-01-01", "2022-12-31"),
        }
        s_start, s_end = stress_ranges[stress_option]
        stress_returns = all_returns_full.loc[s_start:s_end]
        if len(stress_returns) < 5:
            st.warning("ข้อมูลย้อนหลังที่มีไม่ครอบคลุมช่วงนี้ — ลองเพิ่ม 'ข้อมูลย้อนหลังกี่ปี' ในตั้งค่าขั้นสูง")
        else:
            stress_rows = []
            for name, w in last_weights.items():
                total, ar, av, sh, mdd = evaluate(w, stress_returns)
                stress_rows.append({"กลยุทธ์": name, "ผลตอบแทนรวมช่วงนี้": f"{total:.1%}", "Max Drawdown ช่วงนี้": f"{mdd:.1%}"})
            st.dataframe(pd.DataFrame(stress_rows), use_container_width=True, hide_index=True)

# ================= TAB 3: วิเคราะห์เชิงลึก =================
with tab3:
    st.markdown("#### ช่วงความเชื่อมั่น 95% และนัยสำคัญทางสถิติ (Paired t-test & Bootstrap)")
    fig1, ax1 = plt.subplots(figsize=(7, 3.8), facecolor='#0f172a')
    ax1.set_facecolor('#1e293b')
    means, errs = [], []
    for s in strategies:
        vals = df[df.strategy == s]["sharpe"].dropna().values
        mean, sd = vals.mean(), vals.std(ddof=1)
        se = sd / np.sqrt(len(vals))
        t_crit = stats.t.ppf(0.975, df=len(vals) - 1)
        means.append(mean)
        errs.append(t_crit * se)
        
    colors_bar = ["#38bdf8", "#facc15", "#4ade80", "#c084fc"][:len(strategies)]
    ax1.bar([s.split(" ")[0] for s in strategies], means, yerr=errs, capsize=8,
            color=colors_bar, alpha=0.85, ecolor='white')
    ax1.axhline(0, color="gray", linewidth=0.8)
    ax1.set_ylabel(f"Sharpe ratio (mean of {n_folds} folds)", color='#94a3b8')
    ax1.set_title("Sharpe Ratio with 95% Confidence Interval", color='white', fontweight='bold')
    ax1.tick_params(colors='#cbd5e1')
    ax1.grid(True, linestyle='--', alpha=0.2)
    st.pyplot(fig1)

    n_comparisons = len(strategies) * (len(strategies) - 1) // 2
    bonferroni_alpha = 0.05 / n_comparisons
    rows = []
    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            s1, s2 = strategies[i], strategies[j]
            diff = (wide[s1] - wide[s2]).dropna()
            t_stat, p_val = stats.ttest_rel(wide[s1], wide[s2])
            p_boot = block_bootstrap_pvalue(diff.values, block_size=4, n_boot=2000)
            rows.append({
                "คู่เปรียบเทียบ": f"{s1.split(' ')[0]} vs {s2.split(' ')[0]}",
                "p-value (t-test)": round(p_val, 4),
                "p-value (block bootstrap)": round(p_boot, 4) if not np.isnan(p_boot) else "N/A",
                "ผ่านเกณฑ์ Bonferroni?": "ผ่าน" if p_val < bonferroni_alpha else "ไม่ผ่าน",
                "สรุป (เกณฑ์ปกติ p<0.05)": "ต่างกันจริง" if p_val < 0.05 else "ยังสรุปไม่ได้ชัดเจน",
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(f"เทียบทั้งหมด {n_comparisons} คู่พร้อมกัน ใช้เกณฑ์ Bonferroni (α = 0.05/{n_comparisons} ≈ {bonferroni_alpha:.4f}) เพื่อป้องกันการฟลุ๊คจากการเทียบหลายคู่")

    if avg_delta > 0:
        st.markdown("#### กลยุทธ์ D: Markowitz-Shrinkage ลดสัญญาณรบกวนได้แค่ไหน")
        st.caption(
            f"ค่าเฉลี่ยสัดส่วนการ 'หด' (shrinkage intensity) ที่สูตรเลือกเองตลอด {n_folds} รอบ: **{avg_delta:.1%}** "
            "(0% = ไม่หดเลย, 100% = หดเต็มที่) ค่ายิ่งสูง แปลว่าข้อมูลมี Noise เยอะ สูตรเลยลดความผันผวนของน้ำหนักพอร์ตลง"
        )

    st.markdown("#### ใครนำอยู่ ณ จุดไหนของการทดสอบ")
    fig_tally = win_tally_fig(df, strategies)
    st.pyplot(fig_tally)

    st.markdown("#### Correlation Matrix ระหว่างหุ้นที่เลือก")
    fig_corr = correlation_matrix_fig(all_returns_full)
    st.pyplot(fig_corr)

    st.markdown("#### เส้นพรมแดนประสิทธิภาพ (Efficient Frontier) จากการจำลอง Monte Carlo")
    fig_ef = efficient_frontier_fig(all_returns_full, last_weights)
    st.pyplot(fig_ef)

    if "B: Markowitz (สมการคณิตศาสตร์)" in weight_history:
        st.markdown("#### สัดส่วนเปลี่ยนไปยังไงในแต่ละรอบ: Markowitz ปกติ เทียบกับ Shrinkage")
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            fig_wt = weight_evolution_fig(weight_history["B: Markowitz (สมการคณิตศาสตร์)"], selected, "Markowitz (B)")
            st.pyplot(fig_wt)
        with col_w2:
            if "D: Markowitz-Shrinkage (ลดสัญญาณรบกวน)" in weight_history:
                fig_wt_d = weight_evolution_fig(weight_history["D: Markowitz-Shrinkage (ลดสัญญาณรบกวน)"], selected, "Shrinkage (D)")
                st.pyplot(fig_wt_d)

# ================= TAB 4: แชร์ผลลัพธ์ (Social Sharing) =================
with tab4:
    st.subheader("📣 แชร์สรุปผลลัพธ์พอร์ตของคุณ")
    st.write("ดาวน์โหลดภาพการ์ดสรุปผลการทดสอบ เพื่อนำไปแบ่งปันหรือโพสต์ต่อบนโซเชียลมีเดีย")
    
    card_buf = create_share_card(best_strategy, best_return, best_sharpe, best_mdd, selected, capital)
    st.image(card_buf, caption="ตัวอย่าง Social Share Card (ภาพความละเอียดสูง)", use_container_width=False, width=600)
    
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
    • **Risk-Free Rate ($R_f$):** กำหนดไว้ที่ $0.0\%$  
    • **Rebalance Cost:** มีการหักค่าธรรมเนียมการซื้อขายตามอัตราที่ผู้ใช้กำหนดทุกครั้งที่มีการปรับพอร์ต  
    • **Market Capitalization:** กลยุทธ์ C ใช้จำนวนหุ้นชำระแล้ว (Shares Outstanding) ปัจจุบัน  

    #### 3. คำเตือนความเสี่ยง (Risk Disclaimer)
    • ผลการจำลองนี้ไม่รวมปัจจัยเรื่อง **Slippage** และภาษี  
    • การทดสอบย้อนหลังเป็นเพียงเครื่องมือช่วยศึกษารูปแบบสถิติในอดีต **ไม่สามารถใช้เป็นสิ่งยืนยันผลตอบแทนในอนาคตได้**
    """)

st.divider()
st.caption("QuantLab Analytics Engine — เครื่องมือจำลองพอร์ตการลงทุนเชิงปริมาณเพื่อการเรียนรู้และวิจัยทางสถิติ")


### 💡 คำแนะนำเพิ่มเติม:
- รูปภาพ Share Card ในแท็บ 4 จะถูกเรนเดอร์ใหม่เป็นแบบ **Clean High-contrast Dark Theme** ที่อ่านง่าย คมชัด และไม่ติดปัญหา `` สี่เหลี่ยมแน่นอนครับ
- แท็บ 3 กลับมาแสดงผลกราฟครบทั้ง 5 มิติ (Conf. Intervals, Win Tally, Correlation Heatmap, Efficient Frontier, และ Weight Stacked Area Charts) ครับ
