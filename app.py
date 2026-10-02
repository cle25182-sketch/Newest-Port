import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
from datetime import datetime, timedelta
import streamlit as st

st.set_page_config(page_title="จัดพอร์ตหุ้นฉบับนักเรียน", page_icon="📈", layout="wide")

RF_RATE = 0.0

# ------------------------------------------------------------------
# ส่วนคำนวณ
# ------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def load_price_data(tickers, years_back):
    import yfinance as yf
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=years_back * 365)).strftime("%Y-%m-%d")
    raw = yf.download(list(tickers), start=start_date, end=end_date, progress=False, auto_adjust=True)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers[0])
    raw = raw.ffill()
    valid = [t for t in tickers if t in raw.columns and raw[t].notna().sum() >= 30]
    return raw[valid].dropna(), valid


@st.cache_data(ttl=3600, show_spinner=False)
def load_benchmark(years_back):
    """ดัชนี SET Index จริง ใช้เป็นเกณฑ์เทียบภายนอก (ไม่ใช่ S&P 500 เพราะหุ้นที่วิเคราะห์เป็นหุ้นไทย)"""
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


@st.cache_data(ttl=3600, show_spinner=False)
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
    """None และ NaN ทั้งคู่ถือว่า 'ไม่มีข้อมูล' -- ป้องกัน NaN แอบหลุดเข้าไปคำนวณ (NaN เป็น truthy ใน Python
    เช็คด้วย `if v` เฉยๆ จะจับ NaN ไม่ได้)"""
    if v is None:
        return None
    try:
        return None if np.isnan(v) else v
    except TypeError:
        return v


def ledoit_wolf_shrinkage(X):
    """Ledoit-Wolf (2004) covariance shrinkage toward a scaled-identity target, derived from scratch
    (not scipy/sklearn's built-in version). X: T x N matrix of returns (not pre-demeaned).

    แนวคิด: ค่าแปรปรวนร่วมที่คำนวณตรงๆ จากข้อมูล (sample covariance, S) มี estimation error สูงเมื่อ
    จำนวนวันข้อมูล (T) ไม่มากพอเทียบกับจำนวนหุ้น (N) -- ยิ่งน้อยยิ่งคลาดเคลื่อน สูตรนี้ "ผสม" S เข้ากับ
    เป้าหมายที่ง่ายกว่าและมี error ต่ำกว่า (F = เมทริกซ์ทแยงมุม) ตามสัดส่วน delta ที่คำนวณให้เหมาะสมที่สุด
    โดยอัตโนมัติ -- ยิ่งข้อมูลมี noise เยอะ สูตรจะเลือกหด (shrink) เข้าหา F มากขึ้นเอง
    """
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

    # กลยุทธ์ D: Markowitz แบบเดียวกับ B เป๊ะ แต่สลับ sample covariance เป็น covariance แบบ shrinkage
    shrunk_cov, delta = ledoit_wolf_shrinkage(train_ret.values)
    opt_shrink = minimize(neg_sharpe, w_equal, args=(mu.values, shrunk_cov),
                           method="SLSQP", bounds=bounds, constraints=constraints)
    w_shrink = opt_shrink.x if opt_shrink.success else w_equal

    weights = {
        "A: Equal-weight (จัดเงินเท่ากัน)": w_equal,
        "B: Markowitz (สมการคณิตศาสตร์)": w_markowitz,
        "D: Markowitz-Shrinkage (ลดสัญญาณรบกวน)": w_shrink,
    }
    if use_marketcap:
        mcap = train_last_price * shares_arr
        weights["C: Market-cap (ตามขนาดบริษัท)"] = mcap / mcap.sum()
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
        # แก้บั๊ก off-by-one: ราคาสุดท้ายที่ "จริงๆ" อยู่ในช่วง train คือ index [start+train_window]
        # ไม่ใช่ [start+train_window-1] (ซึ่งจะเป็นราคาของวันก่อนหน้านั้นแทน)
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


def correlation_heatmap_fig(all_returns, tickers):
    corr = all_returns.corr()
    labels = [t.split(".")[0] for t in tickers]
    fig, ax = plt.subplots(figsize=(5.5 + 0.3 * len(tickers), 5 + 0.3 * len(tickers)))
    im = ax.imshow(corr.values, cmap="RdYlGn_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black" if abs(corr.values[i, j]) < 0.7 else "white")
    plt.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Correlation Matrix of Selected Stocks")
    plt.tight_layout()
    n = len(tickers)
    off_diag_sum = corr.values.sum() - n
    avg_corr = off_diag_sum / (n * n - n) if n > 1 else 0
    return fig, avg_corr


def weight_evolution_fig(weight_history_df, tickers, strategy_label):
    labels = [t.split(".")[0] for t in tickers]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    values = weight_history_df[tickers].values.T * 100
    ax.stackplot(weight_history_df.index, values, labels=labels, alpha=0.85)
    ax.set_xlabel("Walk-forward round")
    ax.set_ylabel("Weight (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Weight allocation over time — {strategy_label}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=min(len(labels), 6), fontsize=8)
    plt.tight_layout()
    return fig


def win_tally_fig(df, strategies):
    wide = df.pivot(index="round", columns="strategy", values="sharpe")
    winner_per_round = wide.idxmax(axis=1)
    cum_wins = pd.DataFrame({s: (winner_per_round == s).cumsum() for s in strategies})
    fig, ax = plt.subplots(figsize=(9, 4))
    for s in strategies:
        ax.plot(cum_wins.index, cum_wins[s], marker="o", markersize=3, linewidth=1.6, label=s.split(" ")[0])
    ax.set_xlabel("Walk-forward round")
    ax.set_ylabel("Cumulative rounds won (highest Sharpe)")
    ax.set_title("Running Win Tally by Round")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def build_text_report(selected, capital, n_folds, summary, best_strategy, significantly_better_than_all, avg_corr, stress_option):
    lines = []
    lines.append("รายงานสรุปผล: การเปรียบเทียบกลยุทธ์การจัดพอร์ตการลงทุน")
    lines.append("=" * 60)
    lines.append(f"หุ้นที่วิเคราะห์: {', '.join(selected)}")
    lines.append(f"เงินลงทุนเริ่มต้นที่ใช้อ้างอิง: {capital:,.0f} บาท")
    lines.append(f"จำนวนรอบ Walk-Forward Validation: {n_folds} รอบ")
    lines.append(f"ค่าสหสัมพันธ์เฉลี่ยระหว่างหุ้นที่เลือก: {avg_corr:.2f}")
    lines.append("")
    lines.append("ผลเฉลี่ยแต่ละกลยุทธ์:")
    for name, row in summary.iterrows():
        lines.append(f"  - {name}: ผลตอบแทน/ปี {row['ann_return']:.1%}, ความผันผวน {row['ann_vol']:.1%}, "
                      f"Sharpe {row['sharpe']:.2f}, Max Drawdown {row['max_drawdown']:.1%}")
    lines.append("")
    verdict = ("ให้ Sharpe ดีที่สุดและต่างจากกลยุทธ์อื่นอย่างมีนัยสำคัญทางสถิติ (p<0.05)"
               if significantly_better_than_all else
               "ให้ Sharpe เฉลี่ยดีที่สุดในการทดสอบนี้ แต่ยังไม่ต่างจากกลยุทธ์อื่นอย่างมีนัยสำคัญทางสถิติ")
    lines.append(f"สรุป: {best_strategy} {verdict}")
    if stress_option != "ไม่ระบุ":
        lines.append(f"ทดสอบเพิ่มเติมเฉพาะช่วงวิกฤต: {stress_option}")
    lines.append("")
    lines.append("ข้อจำกัด: คำนวณจากข้อมูลอดีต (backtest) เท่านั้น ไม่ใช่การรับประกันผลตอบแทนในอนาคต")
    lines.append("risk-free rate = 0, Market-cap ใช้จำนวนหุ้นปัจจุบันประมาณค่าย้อนหลัง, ยังไม่รวมภาษี")
    return "\n".join(lines)




def block_bootstrap_pvalue(diff_values, block_size=4, n_boot=2000, seed=0):
    """ทดสอบว่าค่าเฉลี่ยผลต่าง (เช่น Sharpe ของ B ลบ A ในแต่ละรอบ) ต่างจาก 0 จริงไหม ด้วยวิธี moving
    block bootstrap แทน paired t-test ธรรมดา -- สุ่มเลือกข้อมูลเป็น "ช่วงต่อเนื่อง" (block) แทนที่จะสุ่ม
    ทีละจุดเดี่ยวๆ เพื่อรักษาความสัมพันธ์ระหว่างรอบที่ติดกัน (เพราะ train ของแต่ละรอบทับซ้อนกันถึง 75%
    รอบที่อยู่ติดกันจึงไม่เป็นอิสระจากกันเต็มที่ตามที่ paired t-test ธรรมดาสมมติไว้)"""
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
    boot_centered = boot_means - boot_means.mean()  # จำลองภายใต้สมมติฐานหลักว่าผลต่างจริงๆ คือ 0
    p_value = np.mean(np.abs(boot_centered) >= abs(observed_mean))
    return p_value


def efficient_frontier_fig(all_returns, weights_by_name, n_portfolios=2500):
    mu, cov = all_returns.mean().values, all_returns.cov().values
    n = len(mu)
    rng = np.random.default_rng(0)
    res = np.zeros((n_portfolios, 3))
    for i in range(n_portfolios):
        w = rng.random(n)
        w /= w.sum()
        port_ret = w @ mu * 252
        port_vol = np.sqrt(max(w @ cov @ w, 0)) * np.sqrt(252)
        res[i] = [port_vol, port_ret, port_ret / port_vol if port_vol > 1e-10 else 0]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sc = ax.scatter(res[:, 0] * 100, res[:, 1] * 100, c=res[:, 2], cmap="viridis", s=6, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="Sharpe ratio")

    markers = list("o*D^v<>")
    for idx, (name, w) in enumerate(weights_by_name.items()):
        port_ret = w @ mu * 252
        port_vol = np.sqrt(max(w @ cov @ w, 0)) * np.sqrt(252)
        ax.scatter(port_vol * 100, port_ret * 100, marker=markers[idx % len(markers)], s=200,
                   edgecolor="black", linewidth=1.3, label=name.split(" ")[0], zorder=5)

    ax.set_xlabel("Volatility / annualized (%)")
    ax.set_ylabel("Expected Return / annualized (%)")
    ax.set_title("Efficient Frontier")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    return fig

# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------

st.title("📈 เว็บแอปจัดพอร์ตหุ้นฉบับนักเรียน")
st.caption(
    "เปรียบเทียบ 3 วิธีแบ่งเงินลงทุนในหุ้น: แบ่งเท่ากัน (Equal-weight), "
    "คำนวณสัดส่วนที่เหมาะสมที่สุดด้วยสูตร Markowitz, และถ่วงน้ำหนักตามมูลค่าบริษัท (Market-cap) "
    "ทดสอบด้วยข้อมูลราคาหุ้นจริงย้อนหลัง แบบ walk-forward validation"
)

with st.expander("📖 ไม่รู้จักศัพท์พวกนี้เลย? เปิดอ่านสรุปสั้นๆ ก่อนได้ที่นี่"):
    st.markdown(
        "- **Sharpe ratio** — ผลตอบแทนเทียบกับความเสี่ยง ยิ่งสูงยิ่งคุ้มค่าความเสี่ยงที่แบกอยู่\n"
        "- **Markowitz Optimization** — สูตรคณิตศาสตร์หาสัดส่วนการลงทุนที่ให้ Sharpe ratio สูงสุด\n"
        "- **Walk-Forward Validation** — ทดสอบหลายรอบด้วยการเลื่อนช่วงข้อมูล train/test ไปเรื่อยๆ แทนแบ่งครั้งเดียว "
        "เพื่อดูว่าผลลัพธ์เชื่อถือได้แค่ไหน ไม่ใช่แค่โชคของช่วงเวลาหนึ่ง\n"
        "- **Max Drawdown** — เงินเคยลดลงจากจุดสูงสุดมากที่สุดกี่ % ในช่วงที่ทดสอบ (วัดว่า \"เจ็บที่สุด\" แค่ไหน)\n"
        "- **p-value** — ความน่าจะเป็นที่ผลต่างที่เห็นเกิดจากความบังเอิญล้วนๆ ต่ำกว่า 0.05 ถือว่าไม่น่าจะบังเอิญ\n"
        "- **Correlation** — หุ้น 2 ตัวเคลื่อนไหวไปด้วยกันแค่ไหน (ยิ่งต่ำ ยิ่งกระจายความเสี่ยงได้ผลดี)\n"
        "- **Efficient Frontier** — เส้นพอร์ตที่ให้ผลตอบแทนดีที่สุด ณ ระดับความเสี่ยงหนึ่งๆ"
    )

with st.sidebar:
    st.header("ตั้งค่า")
    capital = st.number_input("เงินลงทุนเริ่มต้น (บาท)", min_value=1000, value=5000, step=500)

    if "ticker_text" not in st.session_state:
        st.session_state.ticker_text = "PTT.BK, CPALL.BK, AOT.BK, KBANK.BK, ADVANC.BK"

    st.caption("ชุดทดสอบด่วน (สลับตลาดดูว่าข้อสรุปยังเหมือนเดิมไหม):")
    preset_col1, preset_col2 = st.columns(2)
    with preset_col1:
        if st.button("🇹🇭 หุ้นไทย", use_container_width=True):
            st.session_state.ticker_text = "PTT.BK, CPALL.BK, AOT.BK, KBANK.BK, ADVANC.BK"
    with preset_col2:
        if st.button("🇺🇸 หุ้นสหรัฐฯ", use_container_width=True):
            st.session_state.ticker_text = "AAPL, MSFT, GOOGL, AMZN, NVDA"

    ticker_input = st.text_input(
        "พิมพ์รหัสหุ้น (คั่นด้วยจุลภาค ,)",
        key="ticker_text",
        help="ใส่รหัสหุ้น เช่น PTT.BK, CPALL.BK สำหรับหุ้นไทย หรือ AAPL, TSLA สำหรับหุ้นสหรัฐฯ — หรือกดปุ่มชุดทดสอบด่วนด้านบน"
    )
    selected = list(dict.fromkeys([t.strip().upper() for t in ticker_input.split(",") if t.strip()]))

    with st.expander("ตั้งค่าขั้นสูง (ไม่บังคับ)"):
        train_window = st.slider("ช่วง train (วันทำการ)", 126, 378, 252, step=21,
                                  help="จำนวนวันย้อนหลังที่ใช้คำนวณสัดส่วนก่อนแต่ละรอบทดสอบ")
        test_window = st.slider("ช่วง test ต่อรอบ (วันทำการ)", 21, 126, 63, step=21,
                                 help="จำนวนวันที่ใช้ทดสอบสัดส่วนที่คำนวณได้ ต่อ 1 รอบ")
        years_back = st.slider("ข้อมูลย้อนหลังกี่ปี", 3, 10, 6)
        cost_pct = st.slider("ค่าธรรมเนียมการซื้อขายต่อการปรับสมดุลพอร์ต (%)", 0.0, 1.0, 0.1, step=0.05,
                              help="หักออกจากมูลค่าพอร์ตทุกครั้งที่ปรับสัดส่วนใหม่ ในกราฟเงินโตสะสม") / 100
        stress_option = st.selectbox(
            "ทดสอบเฉพาะช่วงวิกฤต (ไม่บังคับ)",
            ["ไม่ระบุ", "COVID-19 (ก.พ.–เม.ย. 2563)", "สงครามการค้าจีน-สหรัฐฯ (ม.ค.–ธ.ค. 2561)",
             "เงินเฟ้อ/ดอกเบี้ยขาขึ้น (ม.ค.–ธ.ค. 2565)"],
            help="ประเมินด้วยสัดส่วนล่าสุดที่คำนวณได้ ว่าถ้าเจอเฉพาะช่วงนี้ผลจะเป็นอย่างไร",
        )

    run = st.button("🚀 เริ่มวิเคราะห์", type="primary", use_container_width=True)
    st.caption("ข้อมูลราคาหุ้นดึงสดจาก Yahoo Finance ทุกครั้งที่กดรัน (แคชไว้ 1 ชั่วโมง)")

if not run:
    st.info("ตั้งค่าทางซ้าย แล้วกด **เริ่มวิเคราะห์** เพื่อดูผลเปรียบเทียบ")
    st.stop()

if len(selected) < 2:
    st.error("กรุณาเลือกหุ้นอย่างน้อย 2 ตัว")
    st.stop()

with st.spinner("กำลังดึงราคาหุ้นย้อนหลัง..."):
    try:
        data, valid_tickers = load_price_data(tuple(selected), years_back)
    except Exception as e:
        st.error(f"ดึงข้อมูลราคาหุ้นไม่สำเร็จ: {e}")
        st.stop()

missing = [t for t in selected if t not in valid_tickers]
if missing:
    st.warning(f"หารหัสหุ้นนี้ไม่เจอ หรือข้อมูลไม่พอ เลยตัดออก: {', '.join(missing)}")
selected = valid_tickers

if len(selected) < 2:
    st.error("เหลือหุ้นที่ใช้ได้น้อยกว่า 2 ตัว กรุณาตรวจสอบรหัสหุ้นแล้วลองใหม่")
    st.stop()

if data.empty or len(data) < train_window + test_window * 3:
    st.error("ข้อมูลย้อนหลังไม่พอสำหรับตั้งค่านี้ ลองลดช่วง train/test หรือเพิ่มจำนวนปีย้อนหลังในตั้งค่าขั้นสูงดู")
    st.stop()

with st.spinner("กำลังตรวจสอบมูลค่าตลาด (สำหรับกลยุทธ์ Market-cap)..."):
    caps_raw = load_market_caps(tuple(selected))
    clean_caps = {t: _clean_shares(caps_raw.get(t)) for t in selected}
    use_marketcap = all(clean_caps[t] is not None for t in selected)
    if not use_marketcap:
        st.warning("ดึงข้อมูลมูลค่าตลาดของบางบริษัทไม่สำเร็จ — แสดงผลเฉพาะกลยุทธ์ Equal-weight และ Markowitz")
    shares_arr = np.array([clean_caps[t] or 0 for t in selected])

with st.spinner("กำลังรัน walk-forward validation..."):
    df, n_folds, n_failed, daily_returns, last_weights, weight_history, avg_delta = run_walk_forward(
        data, shares_arr, use_marketcap, train_window, test_window
    )

if df.empty:
    st.error("ไม่สามารถรันได้ครบแม้แต่ 1 รอบ ลองลดค่า train/test window ในตั้งค่าขั้นสูงดู")
    st.stop()

strategies = list(df["strategy"].unique())
st.success(f"วิเคราะห์เสร็จแล้ว — ทดสอบทั้งหมด {n_folds} รอบ (walk-forward validation)")
if n_failed:
    st.caption(f"หมายเหตุ: การหาค่าเหมาะสมที่สุดไม่ลู่เข้าใน {n_failed}/{n_folds} รอบ (ใช้ equal-weight แทนในรอบนั้น)")

summary = df.groupby("strategy")[["ann_return", "ann_vol", "sharpe", "max_drawdown"]].mean().reindex(strategies)
best_strategy = summary["sharpe"].idxmax()
best_return = summary.loc[best_strategy, "ann_return"]
best_vol = summary.loc[best_strategy, "ann_vol"]
best_sharpe = summary.loc[best_strategy, "sharpe"]
best_mdd = summary.loc[best_strategy, "max_drawdown"]

wide = df.pivot(index="round", columns="strategy", values="sharpe")
significantly_better_than_all = True
for other in strategies:
    if other == best_strategy:
        continue
    _, p_val = stats.ttest_rel(wide[best_strategy], wide[other])
    if p_val >= 0.05:
        significantly_better_than_all = False
        break

all_returns_full = data.pct_change().dropna()
fig_corr, avg_corr = correlation_heatmap_fig(all_returns_full, selected)

tab1, tab2, tab3, tab4 = st.tabs(["📊 สรุปผล", "📈 การเติบโตของเงินทุน", "🔬 วิเคราะห์เชิงลึก", "📥 ดาวน์โหลด"])

# ================= TAB 1: สรุปผล =================
with tab1:
    st.subheader("สรุปผล: กลยุทธ์ที่ Sharpe ดีที่สุดในการทดสอบนี้")
    if significantly_better_than_all:
        st.success(
            f"**{best_strategy}** ให้ Sharpe ratio ดีที่สุด และแตกต่างจากกลยุทธ์อื่นทุกตัวอย่างมีนัยสำคัญทางสถิติ (p < 0.05) "
            f"ในการทดสอบ {n_folds} รอบนี้"
        )
    else:
        st.info(
            f"**{best_strategy}** ให้ Sharpe ratio เฉลี่ยสูงสุดในการทดสอบนี้ แต่ยัง**ไม่ต่างจากกลยุทธ์อื่นอย่างมีนัยสำคัญทางสถิติ** "
            f"(ดูตาราง p-value ในแท็บ 'วิเคราะห์เชิงลึก') — ควรตีความว่า \"ยังแยกไม่ออกชัดเจนว่าวิธีไหนดีกว่าจริง\" มากกว่าฟันธงว่าตัวนี้ชนะ"
        )

    st.markdown("#### ตัวอย่างการจัดสรรเงินตามกลยุทธ์นี้")
    winner_weights = last_weights[best_strategy]
    action_plan = []
    for ticker, weight in zip(selected, winner_weights):
        action_plan.append({
            "รหัสหุ้น": ticker,
            "สัดส่วน (%)": f"{weight * 100:.2f}%",
            "จำนวนเงิน (บาท)": f"{capital * weight:,.2f}",
        })
    st.dataframe(pd.DataFrame(action_plan), use_container_width=True, hide_index=True)
    st.caption("ตัวเลขนี้มาจากสัดส่วนของรอบล่าสุดที่คำนวณได้ ใช้เพื่อประกอบการอธิบายวิธีการเท่านั้น ไม่ใช่คำแนะนำการลงทุนจริง")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("ผลตอบแทนเฉลี่ยต่อปี", f"{best_return * 100:.1f}%")
    with col_m2:
        st.metric("ความผันผวนต่อปี", f"{best_vol * 100:.1f}%")
    with col_m3:
        st.metric("Sharpe ratio", f"{best_sharpe:.2f}")
    with col_m4:
        st.metric("Max Drawdown", f"{best_mdd * 100:.1f}%")

    st.markdown("#### ตารางเปรียบเทียบทั้ง 3 กลยุทธ์ (ค่าเฉลี่ยตลอดทุกรอบ)")
    display_summary = summary.copy()
    display_summary["ann_return"] = (display_summary["ann_return"] * 100).round(1).astype(str) + "%"
    display_summary["ann_vol"] = (display_summary["ann_vol"] * 100).round(1).astype(str) + "%"
    display_summary["max_drawdown"] = (display_summary["max_drawdown"] * 100).round(1).astype(str) + "%"
    display_summary["sharpe"] = display_summary["sharpe"].round(2)
    display_summary.columns = ["ผลตอบแทน/ปี", "ความผันผวน/ปี", "Sharpe ratio", "Max Drawdown"]
    st.dataframe(display_summary, use_container_width=True)
    st.caption("ตารางนี้ยังไม่รวมค่าธรรมเนียมการซื้อขาย (ดูผลที่รวมค่าธรรมเนียมแล้วในแท็บ 'การเติบโตของเงินทุน')")

    st.info(
        "**A: Equal-weight** — แบ่งเงินเท่ากันทุกตัว  \n"
        "**B: Markowitz** — คำนวณสัดส่วนด้วยสูตรคณิตศาสตร์ให้ Sharpe ratio สูงสุด  \n"
        "**C: Market-cap** — ถ่วงน้ำหนักตามมูลค่าบริษัท บริษัทใหญ่กว่าได้สัดส่วนมากกว่า  \n"
        "**D: Markowitz-Shrinkage** — เหมือน B แต่ลด estimation error ของข้อมูลก่อน ด้วยเทคนิค Ledoit-Wolf shrinkage"
    )
    st.caption(f"ค่าสหสัมพันธ์เฉลี่ยระหว่างหุ้นที่เลือก: **{avg_corr:.2f}** (ยิ่งต่ำ ยิ่งกระจายความเสี่ยงได้ผลดี — ดูตารางเต็มในแท็บ 'วิเคราะห์เชิงลึก')")

# ================= TAB 2: การเติบโตของเงินทุน =================
with tab2:
    st.subheader("เงินลงทุนสะสม ถ้าลงทุนต่อเนื่องตลอดช่วงทดสอบ (คิดทบต้น)")
    curves = cumulative_growth(daily_returns, capital, cost_pct, test_window)
    bench_raw = load_benchmark(years_back)

    fig_cum, ax_cum = plt.subplots(figsize=(11, 5))
    for name, curve in curves.items():
        ax_cum.plot(curve.index, curve.values, label=name.split(" ")[0], linewidth=1.6)

    bench_note = ""
    if not bench_raw.empty:
        combined_index = next(iter(curves.values())).index
        bench_aligned = bench_raw.reindex(bench_raw.index.union(combined_index)).ffill().reindex(combined_index)
        if bench_aligned.notna().sum() > 10:
            bench_ret = bench_aligned.pct_change().fillna(0)
            bench_curve = capital * (1 + bench_ret).cumprod()
            ax_cum.plot(bench_curve.index, bench_curve.values, label="Buy & Hold SET Index", linewidth=1.8,
                        linestyle="--", color="black")
        else:
            bench_note = "ข้อมูล SET Index ไม่พอสำหรับช่วงเวลานี้ แสดงเฉพาะ 3 กลยุทธ์"
    else:
        bench_note = "ดึงข้อมูล SET Index (^SET.BK) ไม่สำเร็จ แสดงเฉพาะ 3 กลยุทธ์"

    ax_cum.axhline(capital, color="gray", linewidth=0.7, linestyle=":")
    ax_cum.set_xlabel("Date")
    ax_cum.set_ylabel(f"Portfolio value (baht), start = {capital:,.0f}")
    ax_cum.set_title(f"Cumulative growth -- includes {cost_pct*100:.2f}% cost per rebalance")
    ax_cum.legend(loc="upper left", fontsize=9)
    st.pyplot(fig_cum)
    if bench_note:
        st.caption(f"⚠️ {bench_note}")
    st.caption(
        "เส้นนี้คือ 'เงินก้อนเดียวเดินทางต่อเนื่อง' ข้ามทุกไตรมาสจริง เส้นประดำคือ Buy & Hold SET Index "
        "ล้วนๆ ไม่ปรับพอร์ตเลย ใช้เป็นเกณฑ์เทียบจากภายนอกที่เหมาะกับหุ้นไทยกว่าดัชนีต่างประเทศ"
    )

    if stress_option != "ไม่ระบุ":
        st.subheader(f"ทดสอบเฉพาะช่วงวิกฤต: {stress_option}")
        stress_ranges = {
            "COVID-19 (ก.พ.–เม.ย. 2563)": ("2020-02-01", "2020-04-30"),
            "สงครามการค้าจีน-สหรัฐฯ (ม.ค.–ธ.ค. 2561)": ("2018-01-01", "2018-12-31"),
            "เงินเฟ้อ/ดอกเบี้ยขาขึ้น (ม.ค.–ธ.ค. 2565)": ("2022-01-01", "2022-12-31"),
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
            st.caption("ใช้สัดส่วนน้ำหนักจากรอบล่าสุดของแต่ละกลยุทธ์ ประเมินย้อนกลับเฉพาะช่วงวิกฤตที่เลือก (ไม่ใช่ walk-forward เต็มรูปแบบ เพราะช่วงสั้นเกินจะแบ่ง train/test ได้)")

# ================= TAB 3: วิเคราะห์เชิงลึก =================
with tab3:
    st.markdown("#### ช่วงความเชื่อมั่น 95% และนัยสำคัญทางสถิติ (Paired t-test)")
    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    means, errs = [], []
    for s in strategies:
        vals = df[df.strategy == s]["sharpe"].dropna().values
        mean, sd = vals.mean(), vals.std(ddof=1)
        se = sd / np.sqrt(len(vals))
        t_crit = stats.t.ppf(0.975, df=len(vals) - 1)
        means.append(mean)
        errs.append(t_crit * se)
    ax1.bar([s.split(" ")[0] for s in strategies], means, yerr=errs, capsize=8,
            color=["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"][:len(strategies)], alpha=0.85)
    ax1.axhline(0, color="gray", linewidth=0.8)
    ax1.set_ylabel(f"Sharpe ratio (mean of {n_folds} folds)")
    ax1.set_title("Sharpe ratio with 95% confidence interval")
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
                "เปรียบเทียบ": f"{s1.split(' ')[0]} vs {s2.split(' ')[0]}",
                "p-value (t-test)": round(p_val, 3),
                "p-value (block bootstrap)": round(p_boot, 3) if not np.isnan(p_boot) else "N/A",
                "ผ่านเกณฑ์ Bonferroni?": "ผ่าน" if p_val < bonferroni_alpha else "ไม่ผ่าน",
                "สรุป (เกณฑ์ปกติ p<0.05)": "ต่างกันจริง" if p_val < 0.05 else "ยังสรุปไม่ได้ชัดเจน",
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        f"เทียบทั้งหมด {n_comparisons} คู่พร้อมกัน จึงใช้เกณฑ์ Bonferroni ที่เข้มกว่าปกติ "
        f"(α = 0.05/{n_comparisons} ≈ {bonferroni_alpha:.4f}) เพื่อกันไม่ให้สุ่มเจอผลบวกปลอมจากการเทียบหลายคู่ "
        "ส่วน p-value (block bootstrap) คือการทดสอบที่ไม่ยึดสมมติฐานว่าแต่ละรอบเป็นอิสระจากกัน "
        "(เพราะช่วง train ของรอบที่ติดกันทับซ้อนกันถึง 75%) ถ้า t-test กับ block bootstrap ให้ข้อสรุปไปทางเดียวกัน "
        "แปลว่าผลลัพธ์ทนทานต่อข้อสมมตินี้จริง"
    )

    if avg_delta > 0:
        st.markdown("#### กลยุทธ์ D: Markowitz-Shrinkage ลดสัญญาณรบกวนได้แค่ไหน")
        st.caption(
            f"ค่าเฉลี่ยสัดส่วนการ 'หด' (shrinkage intensity) ที่สูตรเลือกเองตลอด {n_folds} รอบ: **{avg_delta:.1%}** "
            "(0% = ไม่หดเลย ใช้ sample covariance ตรงๆ เหมือนกลยุทธ์ B, 100% = หดเข้าหา target เต็มที่) "
            "ค่ายิ่งสูง แปลว่าข้อมูลที่มีมี estimation error เยอะ สูตรเลยเลือกเชื่อค่าที่คำนวณได้ตรงๆ น้อยลง"
        )

    st.markdown("#### ใครนำอยู่ ณ จุดไหนของการทดสอบ")
    fig_tally = win_tally_fig(df, strategies)
    st.pyplot(fig_tally)
    st.caption("นับสะสมว่าถึงรอบนั้น แต่ละกลยุทธ์ชนะ (Sharpe สูงสุดในรอบ) ไปแล้วกี่รอบ — เส้นที่อยู่บนสุดตอนจบ คือกลยุทธ์ที่ชนะบ่อยที่สุดโดยรวม")

    st.markdown("#### Correlation Matrix ระหว่างหุ้นที่เลือก")
    st.pyplot(fig_corr)
    st.caption("ยิ่งค่าใกล้ 1 (แดง) หุ้น 2 ตัวยิ่งเคลื่อนไหวไปด้วยกัน กระจายความเสี่ยงได้ผลน้อย — ยิ่งใกล้ 0 หรือติดลบ (เขียว) กระจายความเสี่ยงได้ผลดีกว่า")

    st.markdown("#### เส้นพรมแดนประสิทธิภาพ (Efficient Frontier) จากหุ้นที่เลือกจริง")
    fig_ef = efficient_frontier_fig(all_returns_full, last_weights)
    st.pyplot(fig_ef)
    st.caption("จุดสีคือพอร์ตสุ่ม 2,500 แบบจากหุ้นที่เลือกจริง สัญลักษณ์ขอบดำคือตำแหน่งของแต่ละกลยุทธ์ (สัดส่วนจากรอบล่าสุด)")

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
        st.caption(
            "ถ้า Shrinkage ได้ผลจริง สัดส่วนฝั่งขวาควรจะ 'กระโดดไปมารุนแรงน้อยกว่า' ฝั่งซ้าย เพราะ estimation error "
            "ถูกลดลงแล้ว ทำให้สูตรไม่แกว่งตามสัญญาณรบกวนในข้อมูลมากเกินไป"
        )

# ================= TAB 4: ดาวน์โหลด =================
with tab4:
    st.markdown("#### ดาวน์โหลดผลลัพธ์")
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ ผลดิบทุกรอบ (CSV)", csv, "walk_forward_results.csv", "text/csv")

    report_text = build_text_report(selected, capital, n_folds, summary, best_strategy,
                                      significantly_better_than_all, avg_corr, stress_option)
    st.download_button("⬇️ รายงานสรุปผล (TXT)", report_text.encode("utf-8-sig"), "summary_report.txt", "text/plain")
    with st.expander("ดูตัวอย่างรายงานก่อนดาวน์โหลด"):
        st.text(report_text)

st.divider()
st.caption(
    "⚠️ ผลลัพธ์ทั้งหมดคำนวณจากข้อมูลราคาหุ้นในอดีต (backtest) เท่านั้น ไม่ใช่การรับประกันผลตอบแทนในอนาคต "
    "ยังไม่รวมภาษี เครื่องมือนี้จัดทำเพื่อการศึกษาในโครงงานวิทยาศาสตร์ ไม่ใช่คำแนะนำการลงทุน\n\n"
    "ข้อสมมติที่ควรรู้: (1) Sharpe ratio คำนวณโดยตั้ง risk-free rate = 0 เพื่อความง่าย "
    "(2) กลยุทธ์ Market-cap ใช้จำนวนหุ้นที่ออกจำหน่ายปัจจุบัน คูณราคาย้อนหลัง เป็นค่าประมาณมูลค่าตลาดในอดีต "
    "ไม่ใช่มูลค่าตลาดจริงในวันนั้น (3) แต่ละรอบ walk-forward ใช้ข้อมูล train ที่ทับซ้อนกันบางส่วน จึงไม่เป็นอิสระจากกันทั้งหมด "
    "ผลการทดสอบนัยสำคัญทางสถิติจึงควรตีความอย่างระมัดระวัง"
)
