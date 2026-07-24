"""
Portfolio Analytics Platform — Retail-Friendly Streamlit Web App
Run locally:   streamlit run app.py
Deploy free:   push to GitHub, then deploy on share.streamlit.io
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from datetime import datetime
import textwrap
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Understand Your Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }
    .verdict-box {
        background-color: #1a2332;
        padding: 22px 26px;
        border-radius: 12px;
        border-left: 5px solid #4f8ef7;
        font-size: 17px;
        line-height: 1.6;
        margin-bottom: 20px;
    }
    .plain-text {
        color: #a0a8b8;
        font-size: 13px;
        margin-top: -8px;
        margin-bottom: 10px;
    }
    .badge-high { background-color: #1e5631; color: #7ee8a0; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
    .badge-medium { background-color: #6b5b1e; color: #f0d878; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
    .badge-low { background-color: #6b1e1e; color: #f08787; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# CACHED DATA HELPERS
# ------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(tickers, benchmark, start, end):
    all_tickers = tickers + [benchmark]
    raw = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"].dropna(how="all").ffill().dropna()
    return prices


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_sectors(tickers):
    sector_map = {}
    for t in tickers:
        try:
            sector_map[t] = yf.Ticker(t).info.get("sector", "Unknown")
        except Exception:
            sector_map[t] = "Unknown"
    return sector_map


# ------------------------------------------------------------
# CORE MATH HELPERS
# ------------------------------------------------------------
def annualize_return(r):
    return (1 + r).prod() ** (252 / len(r)) - 1


def annualize_vol(r):
    return r.std() * np.sqrt(252)


def sharpe(r, rf):
    return (annualize_return(r) - rf) / annualize_vol(r)


def max_drawdown(cum):
    peak = cum.cummax()
    return ((cum - peak) / peak).min()


def drawdown_series(cum):
    return (cum - cum.cummax()) / cum.cummax()


def recovery_days(cum):
    dd = drawdown_series(cum)
    trough = dd.idxmin()
    after = cum.loc[trough:]
    peak_val = cum.cummax().loc[trough]
    recovered = after[after >= peak_val]
    return (recovered.index[0] - trough).days if len(recovered) else None


def rebalanced_returns(returns, weights, freq):
    w = weights.copy()
    port_ret, period = [], None
    for date, row in returns.iterrows():
        if freq == "monthly" and date.month != period:
            w = weights.copy()
            period = date.month
        elif freq == "quarterly" and (date.month - 1) // 3 != period:
            w = weights.copy()
            period = (date.month - 1) // 3
        elif freq == "annual" and date.year != period:
            w = weights.copy()
            period = date.year
        r = (row * w).sum()
        port_ret.append(r)
        if freq == "none":
            w = w * (1 + row) / (1 + r)
    return pd.Series(port_ret, index=returns.index)


def validate_portfolio(df, tol=0.001):
    issues = []
    if abs(df["Weight"].sum() - 1) > tol:
        issues.append(f"Weights sum to {df['Weight'].sum():.2%}, not 100%")
    dupes = df[df["Ticker"].duplicated()]["Ticker"].tolist()
    if dupes:
        issues.append(f"Duplicate tickers: {dupes}")
    if df["Ticker"].isna().any() or df["Weight"].isna().any():
        issues.append("Missing values found")
    return issues


# ------------------------------------------------------------
# PLAIN-LANGUAGE TRANSLATION LAYER
# ------------------------------------------------------------
def plain_cagr(cagr, bench_cagr):
    diff = cagr - bench_cagr
    if diff > 0.02:
        return f"Your portfolio grew faster than the market benchmark by about {diff:.1%} per year."
    elif diff < -0.02:
        return f"Your portfolio grew slower than the market benchmark by about {abs(diff):.1%} per year."
    else:
        return "Your portfolio grew at roughly the same pace as the market benchmark."


def plain_sharpe(sh):
    if sh > 1.0:
        return "For the risk you took, you earned a strong return — good risk-adjusted performance."
    elif sh > 0.5:
        return "For the risk you took, you earned a fair return — not exceptional, not poor."
    elif sh > 0:
        return "You earned a positive return, but you took on a lot of risk for it."
    else:
        return "The risk you took wasn't rewarded — returns didn't compensate for the ups and downs."


def plain_drawdown(dd, recov):
    recov_text = f"and would have taken about {recov} days to recover" if recov else "and had not yet fully recovered by the end of this period"
    return (
        f"If you'd invested at the worst possible time in this window, your money would have "
        f"temporarily fallen by about {abs(dd):.0%} before recovering, {recov_text}."
    )


def plain_diversification(div_ratio, avg_corr):
    if div_ratio > 1.3:
        badge, label = "badge-high", "High"
        text = "Your stocks tend to move independently of each other — genuine diversification benefit."
    elif div_ratio > 1.1:
        badge, label = "badge-medium", "Medium"
        text = "Your stocks move somewhat together — you have some diversification, but not a lot."
    else:
        badge, label = "badge-low", "Low"
        text = "Your stocks tend to move together — holding many of them isn't spreading your risk as much as it looks."
    return badge, label, text


def plain_concentration(max_sector, max_sector_weight):
    if max_sector_weight > 0.4:
        return f"⚠️ Nearly half or more of your portfolio ({max_sector_weight:.0%}) is in {max_sector} — if that sector has a bad year, a large part of your money moves with it."
    elif max_sector_weight > 0.25:
        return f"Your largest sector exposure is {max_sector} at {max_sector_weight:.0%} — meaningful, but not extreme, concentration."
    else:
        return f"Your largest sector exposure is {max_sector} at {max_sector_weight:.0%} — reasonably spread across sectors."


def plain_stress(name, p_loss, b_loss):
    relative = p_loss - b_loss
    if relative > 0.03:
        comp = f"held up better than the market by about {relative:.0%}"
    elif relative < -0.03:
        comp = f"fell harder than the market by about {abs(relative):.0%}"
    else:
        comp = "moved roughly in line with the market"
    return f"During the {name}, your portfolio would have lost about {abs(p_loss):.0%} — it {comp}."


def build_verdict(cagr, bench_cagr, max_sector, max_sector_weight, div_ratio, dd, n_stocks):
    growth_word = "grew faster than" if cagr > bench_cagr + 0.02 else ("grew slower than" if cagr < bench_cagr - 0.02 else "grew roughly in line with")
    div_word = "was spread across genuinely independent stocks" if div_ratio > 1.3 else ("had some diversification" if div_ratio > 1.1 else "moved largely as one block despite holding several stocks")
    return (
        f"Over this period, your {n_stocks}-stock portfolio {growth_word} the market benchmark, "
        f"was concentrated in <b>{max_sector}</b> ({max_sector_weight:.0%} of your money), "
        f"{div_word}, and would have temporarily lost about <b>{abs(dd):.0%}</b> of its value "
        f"at the worst point in this period."
    )


# ------------------------------------------------------------
# PDF REPORT BUILDER
# ------------------------------------------------------------
class ReportPDF(FPDF):
    def __init__(self, client_name):
        super().__init__()
        self.client_name = client_name

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Your Portfolio, Explained", new_x="LMARGIN", new_y="NEXT", align="C")
        self.set_font("Helvetica", "", 9)
        self.cell(0, 6, f"{self.client_name}  |  Generated {datetime.now().strftime('%d %b %Y')}",
                   new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(4)

    def section_title(self, title):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT", fill=True)
        self.ln(2)

    def _pdf_safe(self, text):
        replacements = {
            "\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'",
            "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u20b9": "Rs ",
        }
        for uni, rep in replacements.items():
            text = text.replace(uni, rep)
        return text.encode("latin-1", errors="ignore").decode("latin-1")

    def wrapped_line(self, text, font_size=10, width_chars=95):
        text = self._pdf_safe(text)
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", font_size)
        for line in textwrap.wrap(text, width=width_chars) or [""]:
            self.set_x(self.l_margin)
            self.cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")


def build_pdf_report(client_name, verdict_text, portfolio, summary, plain_sharpe_text, plain_dd_text,
                      div_label, div_text, concentration_text, stress_texts, rupee_now, rupee_projected,
                      chart_paths):
    pdf = ReportPDF(client_name)
    pdf.add_page()

    pdf.section_title("Your Portfolio in Plain English")
    plain_verdict = verdict_text.replace("<b>", "").replace("</b>", "")
    pdf.wrapped_line(plain_verdict, font_size=11)
    pdf.ln(4)

    pdf.section_title("How Much Did You Earn?")
    pdf.wrapped_line(f"Portfolio CAGR: {summary.loc['CAGR', 'Portfolio']:.2%}  |  Benchmark CAGR: {summary.loc['CAGR', 'Benchmark']:.2%}")
    pdf.wrapped_line(f"Rs {rupee_now:,.0f} invested at the start would be worth Rs {rupee_projected:,.0f} today.")
    pdf.ln(2)
    pdf.image(chart_paths["growth"], w=180)

    pdf.add_page()
    pdf.section_title("Was the Risk Worth It?")
    pdf.wrapped_line(plain_sharpe_text)
    pdf.ln(2)
    pdf.section_title("What's the Worst That Could Have Happened?")
    pdf.wrapped_line(plain_dd_text)
    pdf.ln(2)
    pdf.image(chart_paths["drawdown"], w=180)

    pdf.add_page()
    pdf.section_title(f"Is Your Portfolio Diversified? - {div_label}")
    pdf.wrapped_line(div_text)
    pdf.ln(2)
    pdf.wrapped_line(concentration_text)
    pdf.ln(2)
    pdf.image(chart_paths["corr"], w=120)
    pdf.image(chart_paths["sector"], w=120)

    pdf.add_page()
    pdf.section_title("How Would This Have Survived a Market Crash?")
    for t in stress_texts:
        pdf.wrapped_line(t)
    pdf.ln(2)
    pdf.image(chart_paths["montecarlo"], w=180)

    pdf.add_page()
    pdf.section_title("Your Holdings")
    for _, row in portfolio.iterrows():
        pdf.wrapped_line(f"{row['Ticker']:<15} {row['Weight']:.1%}   Sector: {row.get('Sector', 'N/A')}")

    pdf.add_page()
    pdf.section_title("Good to Know")
    limitations = [
        "This is an educational analysis, not personalized investment advice.",
        "Past performance does not guarantee future results.",
        "Does not account for taxes, brokerage, or transaction costs.",
        "Mutual fund/ETF overlap is not analysed in this version.",
        "Simulations assume future patterns resemble the past - real markets can behave differently.",
    ]
    for l in limitations:
        pdf.wrapped_line(f"- {l}", font_size=9)

    return bytes(pdf.output())


# ------------------------------------------------------------
# SIDEBAR — INPUTS
# ------------------------------------------------------------
st.sidebar.title("📊 Your Portfolio")

input_mode = st.sidebar.radio("Portfolio input", ["Use sample portfolio", "Upload CSV/Excel"])

if input_mode == "Upload CSV/Excel":
    uploaded_file = st.sidebar.file_uploader("Upload Ticker + Weight file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            portfolio = pd.read_csv(uploaded_file)
        else:
            portfolio = pd.read_excel(uploaded_file)
        portfolio.columns = [c.strip().title() for c in portfolio.columns]
        portfolio = portfolio[portfolio["Ticker"].astype(str).str.upper() != "TOTAL"]
        portfolio["Weight"] = portfolio["Weight"].astype(str).str.replace("%", "").astype(float)
        if portfolio["Weight"].sum() > 1.5:
            portfolio["Weight"] = portfolio["Weight"] / 100
    else:
        portfolio = None
else:
    portfolio = pd.DataFrame({
        "Ticker": ["RELIANCE.NS", "HDFCBANK.NS", "TCS.NS", "DIXON.NS", "ICICIBANK.NS", "MARUTI.NS"],
        "Weight": [0.20, 0.20, 0.15, 0.10, 0.15, 0.20],
    })

st.sidebar.markdown("---")
client_name = st.sidebar.text_input("Your name", "Investor")
start_date = st.sidebar.date_input("Look back from", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("Look back to", pd.to_datetime("today"))
initial_investment = st.sidebar.number_input("If you'd invested (₹)", value=1_000_000, step=100000)
benchmark = st.sidebar.selectbox("Compare against", ["^NSEI", "^CRSLDX", "^CNXMID"], format_func=lambda x: {
    "^NSEI": "Nifty 50", "^CRSLDX": "Nifty 500", "^CNXMID": "Nifty Midcap"
}.get(x, x))
rebalance = st.sidebar.selectbox("Rebalancing", ["monthly", "quarterly", "annual", "none"])
n_simulations = st.sidebar.slider("Simulation detail", 1000, 8000, 2500, 500,
                                   help="Higher = more precise but slower. 2500 is a good default.")

with st.sidebar.expander("Advanced settings"):
    risk_free_rate = st.slider("Risk-free rate", 0.0, 0.12, 0.065, 0.005, format="%.3f")

run_button = st.sidebar.button("🚀 Understand My Portfolio", type="primary", use_container_width=True)

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
st.title("Understand Your Portfolio")
st.caption("Upload your stock portfolio and get a plain-English breakdown of how it's really performing.")

if portfolio is None:
    st.info("Upload a CSV/Excel file with `Ticker` and `Weight` columns from the sidebar to begin.")
    st.stop()

with st.expander("📋 Your portfolio input", expanded=not run_button):
    st.dataframe(portfolio, use_container_width=True)

issues = validate_portfolio(portfolio)
if issues:
    for i in issues:
        st.warning(i)
else:
    st.success("Portfolio looks good.")

if not run_button:
    st.stop()

# ------------------------------------------------------------
# RUN PIPELINE
# ------------------------------------------------------------
tickers = portfolio["Ticker"].tolist()
weights = portfolio.set_index("Ticker")["Weight"].reindex(tickers)

with st.spinner("Fetching historical prices..."):
    try:
        prices = fetch_prices(tickers, benchmark, str(start_date), str(end_date))
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        st.stop()

if prices.empty or len(prices) < 30:
    st.error("Not enough historical data returned. Check tickers and date range.")
    st.stop()

stock_prices = prices[tickers]
bench_prices = prices[benchmark]
daily_returns = stock_prices.pct_change().dropna()

with st.spinner("Understanding your sectors..."):
    sector_map = fetch_sectors(tickers)
portfolio["Sector"] = portfolio["Ticker"].map(sector_map)

portfolio_returns = rebalanced_returns(daily_returns, weights, rebalance)
bench_returns = bench_prices.pct_change().dropna()
bench_returns, portfolio_returns = bench_returns.align(portfolio_returns, join="inner")

portfolio_value = initial_investment * (1 + portfolio_returns).cumprod()
port_cum = (1 + portfolio_returns).cumprod()
bench_cum = (1 + bench_returns).cumprod()

beta = np.cov(portfolio_returns, bench_returns)[0, 1] / np.var(bench_returns)
recov_days = recovery_days(port_cum)

summary = pd.DataFrame({
    "Portfolio": [annualize_return(portfolio_returns), annualize_vol(portfolio_returns),
                  sharpe(portfolio_returns, risk_free_rate), max_drawdown(port_cum)],
    "Benchmark": [annualize_return(bench_returns), annualize_vol(bench_returns),
                  sharpe(bench_returns, risk_free_rate), max_drawdown(bench_cum)],
}, index=["CAGR", "Volatility", "Sharpe", "Max Drawdown"])

corr = daily_returns.corr()
avg_corr = (corr.sum().sum() - len(corr)) / (len(corr) ** 2 - len(corr)) if len(corr) > 1 else 0
asset_vols = daily_returns.std() * np.sqrt(252)
div_ratio = (weights * asset_vols).sum() / annualize_vol(portfolio_returns)

sector_weights = portfolio.groupby("Sector")["Weight"].sum().sort_values(ascending=False)
max_sector = sector_weights.index[0]
max_sector_weight = sector_weights.iloc[0]

with st.spinner(f"Running {n_simulations:,} future scenarios..."):
    np.random.seed(42)
    mu = daily_returns.mean().values
    cov_matrix = daily_returns.cov().values
    L = np.linalg.cholesky(cov_matrix)
    horizon = 252
    w_arr = weights.values.astype(np.float32)
    z = np.random.normal(size=(n_simulations, horizon, len(tickers))).astype(np.float32)
    correlated_shocks = z @ L.T.astype(np.float32)
    daily_asset_returns = mu.astype(np.float32) + correlated_shocks
    daily_port_returns = daily_asset_returns @ w_arr
    sim_paths = initial_investment * np.cumprod(1 + daily_port_returns, axis=1)
    sim_final_values = sim_paths[:, -1]
    del z, correlated_shocks, daily_asset_returns

p5, p50, p95 = np.percentile(sim_final_values, [5, 50, 95])
prob_loss = (sim_final_values < initial_investment).mean()

stress_periods = {
    "COVID Crash (Feb-Apr 2020)": ("2020-02-01", "2020-04-30"),
    "2022 Correction": ("2022-01-01", "2022-06-30"),
}
stress_results = {}
for name, (s, e) in stress_periods.items():
    mask = (portfolio_returns.index >= s) & (portfolio_returns.index <= e)
    if mask.sum() < 5:
        continue
    p_loss = (1 + portfolio_returns[mask]).prod() - 1
    b_loss = (1 + bench_returns[mask]).prod() - 1
    stress_results[name] = (p_loss, b_loss)

# ------------------------------------------------------------
# PLAIN-LANGUAGE VERDICT
# ------------------------------------------------------------
verdict_text = build_verdict(
    summary.loc["CAGR", "Portfolio"], summary.loc["CAGR", "Benchmark"],
    max_sector, max_sector_weight, div_ratio, summary.loc["Max Drawdown", "Portfolio"], len(tickers)
)

st.markdown(f'<div class="verdict-box">📌 {verdict_text}</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# HOW MUCH DID YOU EARN
# ------------------------------------------------------------
st.markdown("## 💰 How Much Did You Earn?")
c1, c2 = st.columns(2)
with c1:
    st.metric("If you'd invested", f"₹{initial_investment:,.0f}")
with c2:
    st.metric("It would be worth today", f"₹{portfolio_value.iloc[-1]:,.0f}",
               f"{(portfolio_value.iloc[-1] / initial_investment - 1):+.1%}")
st.markdown(f'<p class="plain-text">{plain_cagr(summary.loc["CAGR", "Portfolio"], summary.loc["CAGR", "Benchmark"])}</p>', unsafe_allow_html=True)

fig1, ax1 = plt.subplots(figsize=(10, 4))
pd.DataFrame({"Your Portfolio": port_cum * initial_investment, "Market Benchmark": bench_cum * initial_investment}).plot(ax=ax1)
ax1.set_title("Growth of Your Investment Over Time")
ax1.set_ylabel("Value (₹)")
st.pyplot(fig1)
fig1.savefig("/tmp/chart_growth.png", dpi=150)

# ------------------------------------------------------------
# WAS THE RISK WORTH IT
# ------------------------------------------------------------
st.markdown("## ⚖️ Was the Risk Worth It?")
st.write(plain_sharpe(summary.loc["Sharpe", "Portfolio"]))
dd_text = plain_drawdown(summary.loc["Max Drawdown", "Portfolio"], recov_days)
st.write(dd_text)

fig3, ax3 = plt.subplots(figsize=(10, 3))
drawdown_series(port_cum).plot(ax=ax3, color="#e05c5c")
ax3.set_title("How Far Below Its Peak Your Portfolio Fell, Over Time")
ax3.set_ylabel("% Below Peak")
st.pyplot(fig3)
fig3.savefig("/tmp/chart_drawdown.png", dpi=150)

with st.expander("See the technical numbers behind this"):
    st.dataframe(summary.style.format("{:.2%}"), use_container_width=True)
    st.write(f"Beta: {beta:.2f} — your portfolio moves about {beta:.1f}x as much as the market on average.")

# ------------------------------------------------------------
# DIVERSIFICATION
# ------------------------------------------------------------
st.markdown("## 🧩 Is Your Portfolio Diversified?")
badge_class, div_label, div_text = plain_diversification(div_ratio, avg_corr)
st.markdown(f'<span class="{badge_class}">{div_label} Diversification</span>', unsafe_allow_html=True)
st.write(div_text)
concentration_text = plain_concentration(max_sector, max_sector_weight)
st.write(concentration_text)

col1, col2 = st.columns(2)
with col1:
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax2)
    ax2.set_title("Which of Your Stocks Move Together")
    st.pyplot(fig2)
    fig2.savefig("/tmp/chart_corr.png", dpi=150)
with col2:
    fig6, ax6 = plt.subplots(figsize=(6, 5))
    sector_weights.plot(kind="barh", ax=ax6, color="steelblue")
    ax6.set_title("Where Your Money Is Invested (By Sector)")
    st.pyplot(fig6)
    fig6.savefig("/tmp/chart_sector.png", dpi=150)

# ------------------------------------------------------------
# STRESS TEST
# ------------------------------------------------------------
st.markdown("## 📉 How Would This Have Survived a Market Crash?")
stress_texts = []
for name, (p_loss, b_loss) in stress_results.items():
    t = plain_stress(name, p_loss, b_loss)
    stress_texts.append(t)
    st.write(f"**{name}**: {t}")

# ------------------------------------------------------------
# FUTURE SCENARIOS (Monte Carlo, plain framing)
# ------------------------------------------------------------
st.markdown("## 🔮 What Might Happen Next Year?")
st.write(
    f"Based on how your stocks have historically moved, if you ran this forward one year "
    f"{n_simulations:,} different ways: a typical outcome would leave you with about "
    f"**₹{p50:,.0f}**, a bad-luck outcome around **₹{p5:,.0f}**, and a good-luck outcome around "
    f"**₹{p95:,.0f}**. There's roughly a **{prob_loss:.0%} chance** you'd end the year with less than you started."
)

fig7, ax7 = plt.subplots(figsize=(10, 5))
pct = np.percentile(sim_paths, [5, 25, 50, 75, 95], axis=0)
days = np.arange(horizon)
ax7.fill_between(days, pct[0], pct[4], alpha=0.15, color="steelblue", label="Unlikely range")
ax7.fill_between(days, pct[1], pct[3], alpha=0.3, color="steelblue", label="Likely range")
ax7.plot(days, pct[2], color="navy", linewidth=2, label="Typical path")
ax7.axhline(initial_investment, color="red", linestyle="--", alpha=0.5, label="What you started with")
ax7.legend()
ax7.set_title("Possible Paths for Your Portfolio Over the Next Year")
ax7.set_ylabel("Value (₹)")
st.pyplot(fig7)
fig7.savefig("/tmp/chart_montecarlo.png", dpi=150)

# ------------------------------------------------------------
# PDF DOWNLOAD
# ------------------------------------------------------------
st.markdown("## 📄 Download Your Report")
chart_paths = {
    "growth": "/tmp/chart_growth.png", "drawdown": "/tmp/chart_drawdown.png",
    "corr": "/tmp/chart_corr.png", "sector": "/tmp/chart_sector.png",
    "montecarlo": "/tmp/chart_montecarlo.png",
}
pdf_bytes = build_pdf_report(
    client_name, verdict_text, portfolio, summary,
    plain_sharpe(summary.loc["Sharpe", "Portfolio"]), dd_text,
    div_label, div_text, concentration_text, stress_texts,
    initial_investment, portfolio_value.iloc[-1], chart_paths,
)
st.download_button(
    "📄 Download My Portfolio Report (PDF)", data=pdf_bytes,
    file_name=f"My_Portfolio_Report_{client_name.replace(' ', '_')}.pdf",
    mime="application/pdf", type="primary",
)

st.caption(
    "This is an educational tool, not investment advice. It does not account for taxes, "
    "brokerage, or transaction costs. Past performance does not guarantee future results."
)
