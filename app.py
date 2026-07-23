"""
Portfolio Analytics Platform — Streamlit Web App
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
import io
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Portfolio Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }
    .metric-card {
        background-color: #1a1d24;
        padding: 16px;
        border-radius: 10px;
        border: 1px solid #2a2d34;
    }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# HELPERS (cached so the app stays fast on repeat runs)
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
# PDF REPORT BUILDER
# ------------------------------------------------------------
class ReportPDF(FPDF):
    def __init__(self, client_name):
        super().__init__()
        self.client_name = client_name

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Portfolio Analysis Report", new_x="LMARGIN", new_y="NEXT", align="C")
        self.set_font("Helvetica", "", 9)
        self.cell(
            0, 6, f"{self.client_name}  |  Generated {datetime.now().strftime('%d %b %Y')}",
            new_x="LMARGIN", new_y="NEXT", align="C",
        )
        self.ln(4)

    def section_title(self, title):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT", fill=True)
        self.ln(2)

    def wrapped_line(self, text, font_size=10, width_chars=95):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", font_size)
        for line in textwrap.wrap(text, width=width_chars) or [""]:
            self.set_x(self.l_margin)
            self.cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")


def build_pdf_report(client_name, portfolio, summary, beta, alpha, tracking_error, info_ratio,
                      sortino, var_95, cvar_95, calmar, recov_days, avg_corr, div_ratio, hhi,
                      eff_holdings, sorted_w, max_sector, max_sector_weight, stress_results,
                      p50, p5, p95, prob_loss, sim_var_95, sim_cvar_95, n_sims, chart_paths):
    pdf = ReportPDF(client_name)
    pdf.add_page()

    pdf.section_title("1. Portfolio Holdings")
    for _, row in portfolio.iterrows():
        pdf.wrapped_line(f"{row['Ticker']:<15} {row['Weight']:.1%}   Sector: {row.get('Sector', 'N/A')}")
    pdf.ln(3)

    pdf.section_title("2. Performance vs Benchmark")
    for metric in summary.index:
        p, b = summary.loc[metric, "Portfolio"], summary.loc[metric, "Benchmark"]
        pdf.wrapped_line(f"{metric:<15} Portfolio: {p:.2%}   Benchmark: {b:.2%}")
    pdf.wrapped_line(f"Beta: {beta:.2f}  Alpha: {alpha:.2%}  Tracking Error: {tracking_error:.2%}  Info Ratio: {info_ratio:.2f}")
    pdf.ln(3)
    pdf.image(chart_paths["growth"], w=180)

    pdf.add_page()
    pdf.section_title("3. Risk Metrics")
    pdf.wrapped_line(f"Sortino Ratio: {sortino:.2f}")
    pdf.wrapped_line(f"Daily VaR (95%): {var_95:.2%}   CVaR (95%): {cvar_95:.2%}")
    pdf.wrapped_line(f"Calmar Ratio: {calmar:.2f}")
    pdf.wrapped_line(f"Drawdown recovery: {recov_days if recov_days else 'not yet recovered'} days")
    pdf.ln(2)
    pdf.image(chart_paths["drawdown"], w=180)

    pdf.add_page()
    pdf.section_title("4. Diversification & Concentration")
    pdf.wrapped_line(f"Average pairwise correlation: {avg_corr:.2f}   Diversification ratio: {div_ratio:.2f}")
    pdf.wrapped_line(f"HHI: {hhi:.3f}   Effective holdings: {eff_holdings:.1f}")
    pdf.wrapped_line(f"Top 3 concentration: {sorted_w.head(3).sum():.1%}")
    pdf.wrapped_line(f"Largest sector exposure: {max_sector} ({max_sector_weight:.1%})")
    pdf.ln(2)
    pdf.image(chart_paths["corr"], w=120)
    pdf.image(chart_paths["sector"], w=120)
    pdf.image(chart_paths["riskcontrib"], w=150)

    pdf.add_page()
    pdf.section_title("5. Rolling Analysis")
    pdf.image(chart_paths["rolling"], w=180)

    pdf.add_page()
    pdf.section_title("6. Historical Stress Testing")
    for name, (p_loss, b_loss) in stress_results.items():
        pdf.wrapped_line(f"{name}: Portfolio {p_loss:.2%}   Benchmark {b_loss:.2%}   Relative {p_loss - b_loss:+.2%}")

    pdf.add_page()
    pdf.section_title("7. Monte Carlo Forward Simulation (1-Year)")
    pdf.wrapped_line(f"Simulations run: {n_sims:,}")
    pdf.wrapped_line(f"Median projected value: Rs {p50:,.0f}")
    pdf.wrapped_line(f"Downside (5th pct): Rs {p5:,.0f}   Upside (95th pct): Rs {p95:,.0f}")
    pdf.wrapped_line(f"Probability of loss over 1Y: {prob_loss:.1%}")
    pdf.wrapped_line(f"Simulated 1Y VaR(95%): Rs {sim_var_95:,.0f}   CVaR(95%): Rs {sim_cvar_95:,.0f}")
    pdf.ln(2)
    pdf.image(chart_paths["montecarlo"], w=180)

    pdf.add_page()
    pdf.section_title("8. Limitations of This Analysis")
    limitations = [
        "Does not account for taxes, brokerage, or transaction costs.",
        "Assumes adjusted price data fully reflects corporate actions.",
        "Mutual fund/ETF underlying-holding overlap is not analysed in this version.",
        "Monte Carlo assumes historical volatility/correlation persist and returns are normal - understates tail risk.",
        "Does not reflect investor-specific cash flows unless purchase/withdrawal history is supplied.",
    ]
    for l in limitations:
        pdf.wrapped_line(f"- {l}", font_size=9)

    return bytes(pdf.output())


# ------------------------------------------------------------
# SIDEBAR — INPUTS
# ------------------------------------------------------------
st.sidebar.title("📊 Portfolio Setup")

input_mode = st.sidebar.radio("Portfolio input", ["Use sample portfolio", "Upload CSV/Excel"])

if input_mode == "Upload CSV/Excel":
    uploaded_file = st.sidebar.file_uploader("Upload Ticker + Weight file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            portfolio = pd.read_csv(uploaded_file)
        else:
            portfolio = pd.read_excel(uploaded_file)
        portfolio.columns = [c.strip().title() for c in portfolio.columns]
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
client_name = st.sidebar.text_input("Client name", "Sample Client")
start_date = st.sidebar.date_input("Backtest start", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("Backtest end", pd.to_datetime("today"))
initial_investment = st.sidebar.number_input("Initial investment (₹)", value=1_000_000, step=100000)
benchmark = st.sidebar.selectbox("Benchmark", ["^NSEI", "^CRSLDX", "^CNXMID"], format_func=lambda x: {
    "^NSEI": "Nifty 50", "^CRSLDX": "Nifty 500", "^CNXMID": "Nifty Midcap"
}.get(x, x))
rebalance = st.sidebar.selectbox("Rebalancing", ["monthly", "quarterly", "annual", "none"])
risk_free_rate = st.sidebar.slider("Risk-free rate", 0.0, 0.12, 0.065, 0.005, format="%.3f")
n_simulations = st.sidebar.slider("Monte Carlo simulations", 1000, 20000, 10000, 1000)

run_button = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
st.title("Portfolio Analytics Platform")
st.caption("Upload a listed-equity portfolio to get performance, risk, diversification, and stress-test analysis.")

if portfolio is None:
    st.info("Upload a CSV/Excel file with `Ticker` and `Weight` columns from the sidebar to begin.")
    st.stop()

with st.expander("📋 Portfolio input", expanded=not run_button):
    st.dataframe(portfolio, use_container_width=True)

issues = validate_portfolio(portfolio)
if issues:
    for i in issues:
        st.warning(i)
else:
    st.success("Portfolio passes validation.")

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

with st.spinner("Fetching sector data..."):
    sector_map = fetch_sectors(tickers)
portfolio["Sector"] = portfolio["Ticker"].map(sector_map)

portfolio_returns = rebalanced_returns(daily_returns, weights, rebalance)
bench_returns = bench_prices.pct_change().dropna()
bench_returns, portfolio_returns = bench_returns.align(portfolio_returns, join="inner")

portfolio_value = initial_investment * (1 + portfolio_returns).cumprod()
port_cum = (1 + portfolio_returns).cumprod()
bench_cum = (1 + bench_returns).cumprod()

beta = np.cov(portfolio_returns, bench_returns)[0, 1] / np.var(bench_returns)
alpha = annualize_return(portfolio_returns) - (risk_free_rate + beta * (annualize_return(bench_returns) - risk_free_rate))
tracking_error = (portfolio_returns - bench_returns).std() * np.sqrt(252)
info_ratio = (annualize_return(portfolio_returns) - annualize_return(bench_returns)) / tracking_error
recov_days = recovery_days(port_cum)

summary = pd.DataFrame({
    "Portfolio": [annualize_return(portfolio_returns), annualize_vol(portfolio_returns),
                  sharpe(portfolio_returns, risk_free_rate), max_drawdown(port_cum)],
    "Benchmark": [annualize_return(bench_returns), annualize_vol(bench_returns),
                  sharpe(bench_returns, risk_free_rate), max_drawdown(bench_cum)],
}, index=["CAGR", "Volatility", "Sharpe", "Max Drawdown"])

downside = portfolio_returns[portfolio_returns < 0]
sortino = (annualize_return(portfolio_returns) - risk_free_rate) / (downside.std() * np.sqrt(252))
var_95 = np.percentile(portfolio_returns, 5)
cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
calmar = annualize_return(portfolio_returns) / abs(max_drawdown(port_cum))

corr = daily_returns.corr()
avg_corr = (corr.sum().sum() - len(corr)) / (len(corr) ** 2 - len(corr)) if len(corr) > 1 else 0
asset_vols = daily_returns.std() * np.sqrt(252)
div_ratio = (weights * asset_vols).sum() / annualize_vol(portfolio_returns)

sorted_w = weights.sort_values(ascending=False)
hhi = (weights ** 2).sum()
eff_holdings = 1 / hhi

sector_weights = portfolio.groupby("Sector")["Weight"].sum().sort_values(ascending=False)
max_sector = sector_weights.index[0]
max_sector_weight = sector_weights.iloc[0]

cov = daily_returns.cov() * 252
w_arr = weights.values
port_var = w_arr @ cov.values @ w_arr
marginal_contrib = cov.values @ w_arr
pct_contrib = (w_arr * marginal_contrib) / port_var
risk_contrib = pd.Series(pct_contrib, index=weights.index).sort_values(ascending=False)

window = min(252, len(portfolio_returns) - 1)
rolling_ret = portfolio_returns.rolling(window).apply(lambda x: (1 + x).prod() ** (252 / window) - 1)
rolling_vol = portfolio_returns.rolling(window).std() * np.sqrt(252)
rolling_sharpe = (rolling_ret - risk_free_rate) / rolling_vol

stress_periods = {
    "COVID Crash": ("2020-02-01", "2020-04-30"),
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

with st.spinner(f"Running {n_simulations:,} Monte Carlo simulations..."):
    np.random.seed(42)
    mu = daily_returns.mean().values
    cov_matrix = daily_returns.cov().values
    L = np.linalg.cholesky(cov_matrix)
    horizon = 252
    z = np.random.normal(size=(n_simulations, horizon, len(tickers)))
    correlated_shocks = z @ L.T
    daily_asset_returns = mu + correlated_shocks
    daily_port_returns = daily_asset_returns @ w_arr
    sim_paths = initial_investment * np.cumprod(1 + daily_port_returns, axis=1)
    sim_final_values = sim_paths[:, -1]

p5, p25, p50, p75, p95 = np.percentile(sim_final_values, [5, 25, 50, 75, 95])
prob_loss = (sim_final_values < initial_investment).mean()
sim_var_95 = initial_investment - p5
sim_cvar_95 = initial_investment - sim_final_values[sim_final_values <= p5].mean()

# ------------------------------------------------------------
# DASHBOARD DISPLAY
# ------------------------------------------------------------
st.markdown("## Performance Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("CAGR", f"{summary.loc['CAGR', 'Portfolio']:.2%}", f"{summary.loc['CAGR', 'Portfolio'] - summary.loc['CAGR', 'Benchmark']:+.2%} vs bench")
c2.metric("Sharpe Ratio", f"{summary.loc['Sharpe', 'Portfolio']:.2f}")
c3.metric("Max Drawdown", f"{summary.loc['Max Drawdown', 'Portfolio']:.2%}")
c4.metric("Final Value", f"₹{portfolio_value.iloc[-1]:,.0f}")

fig1, ax1 = plt.subplots(figsize=(10, 4))
pd.DataFrame({"Portfolio": port_cum, "Benchmark": bench_cum}).plot(ax=ax1)
ax1.set_title("Portfolio vs Benchmark (Growth of ₹1)")
st.pyplot(fig1)
fig1.savefig("/tmp/chart_growth.png", dpi=150)

st.markdown("## Risk Metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Sortino", f"{sortino:.2f}")
c2.metric("Daily VaR (95%)", f"{var_95:.2%}")
c3.metric("Daily CVaR (95%)", f"{cvar_95:.2%}")
c4.metric("Calmar", f"{calmar:.2f}")

fig3, ax3 = plt.subplots(figsize=(10, 3))
drawdown_series(port_cum).plot(ax=ax3, color="red")
ax3.set_title("Portfolio Drawdown")
st.pyplot(fig3)
fig3.savefig("/tmp/chart_drawdown.png", dpi=150)

st.markdown("## Diversification & Concentration")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg Correlation", f"{avg_corr:.2f}")
c2.metric("Diversification Ratio", f"{div_ratio:.2f}")
c3.metric("Effective Holdings", f"{eff_holdings:.1f} / {len(weights)}")
c4.metric("Top Sector", f"{max_sector}", f"{max_sector_weight:.1%}")

col1, col2 = st.columns(2)
with col1:
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax2)
    ax2.set_title("Correlation Matrix")
    st.pyplot(fig2)
    fig2.savefig("/tmp/chart_corr.png", dpi=150)
with col2:
    fig6, ax6 = plt.subplots(figsize=(6, 5))
    sector_weights.plot(kind="barh", ax=ax6, color="steelblue")
    ax6.set_title("Sector Exposure")
    st.pyplot(fig6)
    fig6.savefig("/tmp/chart_sector.png", dpi=150)

fig5, ax5 = plt.subplots(figsize=(10, 4))
risk_contrib.sort_values().plot(kind="barh", ax=ax5, color="darkorange")
ax5.set_title("Risk Contribution by Stock")
st.pyplot(fig5)
fig5.savefig("/tmp/chart_riskcontrib.png", dpi=150)

st.markdown("## Rolling Analysis (1Y window)")
fig4, axes4 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
rolling_ret.plot(ax=axes4[0], title="Rolling Return")
rolling_vol.plot(ax=axes4[1], title="Rolling Volatility")
rolling_sharpe.plot(ax=axes4[2], title="Rolling Sharpe")
fig4.tight_layout()
st.pyplot(fig4)
fig4.savefig("/tmp/chart_rolling.png", dpi=150)

st.markdown("## Historical Stress Testing")
for name, (p_loss, b_loss) in stress_results.items():
    st.write(f"**{name}**: Portfolio `{p_loss:.2%}` | Benchmark `{b_loss:.2%}` | Relative `{p_loss - b_loss:+.2%}`")

st.markdown(f"## Monte Carlo Simulation ({n_simulations:,} paths, 1-year forward)")
c1, c2, c3 = st.columns(3)
c1.metric("Median Outcome", f"₹{p50:,.0f}")
c2.metric("Probability of Loss", f"{prob_loss:.1%}")
c3.metric("1Y VaR (95%)", f"₹{sim_var_95:,.0f}")

fig7, ax7 = plt.subplots(figsize=(10, 5))
pct = np.percentile(sim_paths, [5, 25, 50, 75, 95], axis=0)
days = np.arange(horizon)
ax7.fill_between(days, pct[0], pct[4], alpha=0.15, color="steelblue", label="5th-95th pct")
ax7.fill_between(days, pct[1], pct[3], alpha=0.3, color="steelblue", label="25th-75th pct")
ax7.plot(days, pct[2], color="navy", linewidth=2, label="Median path")
ax7.axhline(initial_investment, color="red", linestyle="--", alpha=0.5, label="Initial investment")
ax7.legend()
ax7.set_title("Monte Carlo Fan Chart")
st.pyplot(fig7)
fig7.savefig("/tmp/chart_montecarlo.png", dpi=150)

# ------------------------------------------------------------
# PDF DOWNLOAD
# ------------------------------------------------------------
st.markdown("## Download Report")
chart_paths = {
    "growth": "/tmp/chart_growth.png", "drawdown": "/tmp/chart_drawdown.png",
    "corr": "/tmp/chart_corr.png", "sector": "/tmp/chart_sector.png",
    "riskcontrib": "/tmp/chart_riskcontrib.png", "rolling": "/tmp/chart_rolling.png",
    "montecarlo": "/tmp/chart_montecarlo.png",
}
pdf_bytes = build_pdf_report(
    client_name, portfolio, summary, beta, alpha, tracking_error, info_ratio,
    sortino, var_95, cvar_95, calmar, recov_days, avg_corr, div_ratio, hhi,
    eff_holdings, sorted_w, max_sector, max_sector_weight, stress_results,
    p50, p5, p95, prob_loss, sim_var_95, sim_cvar_95, n_simulations, chart_paths,
)
st.download_button(
    "📄 Download Full PDF Report", data=pdf_bytes,
    file_name=f"Portfolio_Report_{client_name.replace(' ', '_')}.pdf",
    mime="application/pdf", type="primary",
)

st.caption(
    "Disclaimer: This is an analytical tool, not investment advice. Does not account for taxes, "
    "brokerage, or transaction costs. Past performance does not guarantee future results."
)
