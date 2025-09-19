"""
CFO Helper - Streamlit app (single-file)
Features:
 - Load sample financials or upload CSV
 - Sliders for hires, marketing, price change
 - Run simulation (billed per scenario)
 - Export PDF report (billed per report)
 - Usage counters + transactions log (Flexprice simulation)
 - Simulated Pathway update button (pulls mock fresh data)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO
import datetime
import json
import os

# ---------------------------
# Helper: default sample data
# ---------------------------
def get_sample_financials():
    # Monthly baseline (example small startup)
    # Columns: month, revenue, expenses, cash_balance (end of month)
    months = pd.date_range(start="2025-01-01", periods=12, freq="M")
    revenue = np.array([200000, 220000, 210000, 230000, 240000, 250000, 260000, 270000, 280000, 290000, 300000, 310000])
    fixed_expenses = 80000  # monthly fixed
    hiring_cost_per_engineer_month = 40000  # monthly loaded cost per hire
    marketing = np.array([10000]*12)
    other = np.array([20000]*12)
    expenses = fixed_expenses + marketing + other
    cash_start = 1_000_000
    cash_balances = []
    cash = cash_start
    for i in range(12):
        profit = revenue[i] - expenses[i]
        cash += profit
        cash_balances.append(cash)
    df = pd.DataFrame({
        "month": months,
        "revenue": revenue,
        "expenses": expenses,
        "cash_balance": cash_balances
    })
    meta = {
        "cash_start": cash_start,
        "fixed_expenses_monthly": fixed_expenses,
        "hire_cost_monthly": hiring_cost_per_engineer_month,
        "current_headcount": 5,
        "average_price_per_unit": 100,  # not used directly but available
    }
    return df, meta

# ---------------------------------
# Simulation engine
# ---------------------------------
def run_simulation(financial_df, meta, hires_delta, marketing_delta, price_change_pct, months_out=12):
    """
    financial_df: baseline monthly dataframe (must have revenue, expenses, cash_balance)
    meta: metadata dict with hire_cost_monthly and current_headcount
    hires_delta: int (can be negative)
    marketing_delta: additional monthly marketing spend (absolute ₹)
    price_change_pct: percentage change in price (positive or negative)
    """
    df = financial_df.copy().reset_index(drop=True)
    # assumptions:
    # - Price change affects revenue proportionally (we simulate a linear effect)
    # - New hires add monthly cost immediately
    hire_cost = meta.get("hire_cost_monthly", 40000)
    original_headcount = meta.get("current_headcount", 5)
    new_headcount = original_headcount + hires_delta

    # Calculate modified revenue and expenses
    # Simulate revenue change: scale revenue by (1 + price_change_pct/100)
    df["sim_revenue"] = df["revenue"] * (1 + price_change_pct/100.0)
    # Simulate expenses: add marketing_delta and hires cost delta
    df["sim_expenses"] = df["expenses"] + marketing_delta + max(hires_delta,0)*hire_cost
    # If hires_delta negative -> assume cost reduction in future months
    if hires_delta < 0:
        df["sim_expenses"] = df["expenses"] + marketing_delta + hires_delta*hire_cost  # negative reduces cost

    # Compute profit and cash runway projection
    # Start from current cash (last cash balance)
    current_cash = df["cash_balance"].iloc[-1]
    sim_cash = []
    cash = current_cash
    for i in range(months_out):
        # for months beyond given months, repeat last month pattern
        if i < len(df):
            rev = df.loc[i, "sim_revenue"]
            exp = df.loc[i, "sim_expenses"]
        else:
            rev = df["sim_revenue"].iloc[-1]
            exp = df["sim_expenses"].iloc[-1]
        profit = rev - exp
        cash += profit
        sim_cash.append(cash)

    # compute runway: how many months until cash <= 0 assuming monthly net burn average
    monthly_net = (df["sim_revenue"] - df["sim_expenses"]).mean()
    if monthly_net >= 0:
        runway_months = float('inf')  # growing cash
    else:
        runway_months = max(0.0, current_cash / (-monthly_net))

    summary = {
        "current_cash": float(current_cash),
        "projected_cash_after_months": float(sim_cash[-1]),
        "runway_months": float(runway_months),
        "new_headcount": int(new_headcount),
        "monthly_net": float(monthly_net),
    }

    # produce a small timeseries for plotting
    ts = pd.DataFrame({
        "month_index": list(range(1, months_out+1)),
        "projected_cash": sim_cash
    })

    return df, ts, summary

# ---------------------------
# Simple Flexprice simulator
# ---------------------------
FLEXPRICE_FILE = "flexprice_state.json"

def load_state():
    if "cfo_state" in st.session_state:
        return st.session_state["cfo_state"]
    # load from file if exists
    if os.path.exists(FLEXPRICE_FILE):
        try:
            with open(FLEXPRICE_FILE, "r") as f:
                st.session_state["cfo_state"] = json.load(f)
        except Exception:
            st.session_state["cfo_state"] = default_state()
    else:
        st.session_state["cfo_state"] = default_state()
    return st.session_state["cfo_state"]

def save_state():
    state = st.session_state.get("cfo_state", default_state())
    with open(FLEXPRICE_FILE, "w") as f:
        json.dump(state, f, default=str)

def default_state():
    return {
        "credits": 100,  # demo starting credits
        "scenarios_run": 0,
        "reports_generated": 0,
        "transactions": []  # list of dicts: {time, type, amount, desc}
    }

def charge(amount, ttype, desc):
    state = load_state()
    state["credits"] -= amount
    if ttype == "scenario":
        state["scenarios_run"] += 1
    elif ttype == "report":
        state["reports_generated"] += 1
    tx = {"time": str(datetime.datetime.now()), "type": ttype, "amount": amount, "desc": desc}
    state["transactions"].insert(0, tx)
    st.session_state["cfo_state"] = state
    save_state()

# ---------------------------
# Report generator (PDF)
# ---------------------------
def generate_pdf_report(summary, params, ts_plot_df, filename="cfo_report.pdf"):
    """
    summary: dict summary from simulation
    params: user inputs metadata
    ts_plot_df: small dataframe with projected cash (month_index, projected_cash)
    returns bytes of PDF
    Uses reportlab to create a simple one-page PDF.
    """
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    left = 40
    top = height - 40
    c.setFont("Helvetica-Bold", 16)
    c.drawString(left, top, "CFO Helper - Scenario Report")
    c.setFont("Helvetica", 10)
    c.drawString(left, top-20, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # input params
    y = top - 50
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Scenario Parameters:")
    c.setFont("Helvetica", 10)
    y -= 16
    for k,v in params.items():
        c.drawString(left+10, y, f"- {k}: {v}")
        y -= 14

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Summary:")
    c.setFont("Helvetica", 10)
    y -= 16
    c.drawString(left+10, y, f"Current cash: ₹{summary['current_cash']:,.2f}")
    y -= 14
    projected_cash = summary["projected_cash_after_months"]
    c.drawString(left+10, y, f"Projected cash after period: ₹{projected_cash:,.2f}")
    y -= 14
    runway = summary["runway_months"]
    if runway == float('inf'):
        runway_text = "Growing cash (no runway limit)"
    else:
        runway_text = f"{runway:.2f} months"
    c.drawString(left+10, y, f"Estimated runway: {runway_text}")
    y -= 20

    # small table of projected cash few months
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Projected cash (first 6 months):")
    y -= 14
    c.setFont("Helvetica", 10)
    for _, row in ts_plot_df.head(6).iterrows():
        c.drawString(left+10, y, f"Month {int(row['month_index'])}: ₹{row['projected_cash']:,.2f}")
        y -= 12

    # footer
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(left, 40, "CFO Helper - Demo report (HackWithHyderabad).")
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()

# ------------------------------------------
# STREAMLIT UI
# ------------------------------------------
st.set_page_config(page_title="CFO Helper", layout="wide")
st.title("CFO Helper — Budget & Runway Simulator (Demo)")

# load or initialize state
state = load_state()

# left column: data / inputs
col1, col2, col3 = st.columns([2,3,2])

with col1:
    st.header("Data")
    if "financial_df" not in st.session_state:
        sample_df, sample_meta = get_sample_financials()
        st.session_state["financial_df"] = sample_df
        st.session_state["meta"] = sample_meta
    # upload option
    uploaded = st.file_uploader("Upload monthly financial CSV (cols: month,revenue,expenses,cash_balance)", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded, parse_dates=["month"])
            st.session_state["financial_df"] = df
            st.success("Uploaded financials loaded.")
        except Exception as e:
            st.error("Failed to load CSV. Make sure it has columns month,revenue,expenses,cash_balance.")
    st.write("Sample financial snapshot:")
    st.dataframe(st.session_state["financial_df"].head(6))

    st.markdown("---")
    st.header("Pathway (Simulated)")
    st.write("Simulate a Pathway update that changes revenue or expenses (mock external data).")
    pathway_choice = st.selectbox("Update type", ["Add new expense", "Increase revenue", "Add irregular expense"])
    pathway_amount = st.number_input("Amount (₹)", value=5000, step=1000)
    if st.button("Simulate Pathway update"):
        # apply a simple change to the baseline data (last 3 months)
        df = st.session_state["financial_df"]
        if pathway_choice == "Add new expense":
            df["expenses"] = df["expenses"] + pathway_amount
            st.success(f"Added ₹{pathway_amount} to expenses across months.")
            desc = f"Pathway: added expense ₹{pathway_amount}"
        elif pathway_choice == "Increase revenue":
            df["revenue"] = df["revenue"] + pathway_amount
            st.success(f"Increased revenue by ₹{pathway_amount} across months.")
            desc = f"Pathway: increased revenue ₹{pathway_amount}"
        else:
            # one-time expense in next month
            df.loc[0, "expenses"] = df.loc[0, "expenses"] + pathway_amount
            st.success(f"Added one-time expense of ₹{pathway_amount} to next month.")
            desc = f"Pathway: one-time expense ₹{pathway_amount}"
        # write back
        st.session_state["financial_df"] = df
        # log transaction (no charge, it's external)
        state = load_state()
        state["transactions"].insert(0, {"time": str(datetime.datetime.now()), "type": "pathway_update", "amount": 0, "desc": desc})
        st.session_state["cfo_state"] = state
        save_state()
        st.experimental_rerun()

with col2:
    st.header("Scenario Inputs")
    hires_delta = st.number_input("Hire (+) / Fire (−) engineers (number)", value=0, step=1)
    marketing_delta = st.number_input("Additional monthly marketing spend (₹)", value=0, step=1000)
    price_change_pct = st.slider("Product price change (%)", min_value=-50, max_value=200, value=0)
    months_out = st.slider("Projection period (months)", min_value=3, max_value=24, value=12)

    st.markdown("**Assumptions**")
    st.write(f"Monthly hire cost per engineer (from metadata): ₹{st.session_state['meta']['hire_cost_monthly']:,}")
    st.write(f"Starting cash balance: ₹{st.session_state['financial_df']['cash_balance'].iloc[-1]:,.2f}")

    if st.button("Run Simulation"):
        # charge per scenario
        SCENARIO_COST = 1  # credits per simulation
        charge(SCENARIO_COST, "scenario", f"Run simulation: hires {hires_delta}, marketing +₹{marketing_delta}, price {price_change_pct}%")
        df, ts, summary = run_simulation(st.session_state["financial_df"], st.session_state["meta"],
                                         hires_delta=int(hires_delta), marketing_delta=float(marketing_delta),
                                         price_change_pct=float(price_change_pct), months_out=int(months_out))
        # store last results
        st.session_state["last_sim"] = {"df": df.to_dict(), "ts": ts.to_dict(), "summary": summary,
                                       "params": {"hires_delta": int(hires_delta), "marketing_delta": float(marketing_delta),
                                                  "price_change_pct": float(price_change_pct), "months_out": int(months_out)}}
        st.experimental_rerun()

    # show last sim if exists
    if "last_sim" in st.session_state and st.session_state["last_sim"] is not None:
        last = st.session_state["last_sim"]
        summary = last["summary"]
        st.subheader("Last Simulation Summary")
        st.metric("Projected cash", f"₹{summary['projected_cash_after_months']:,.2f}")
        runway = summary["runway_months"]
        if runway == float('inf'):
            st.metric("Runway", "Growing")
        else:
            st.metric("Runway (months)", f"{runway:.2f}")
        st.write("Details:")
        st.json(summary)

with col3:
    st.header("Flexprice / Usage")
    state = load_state()
    st.write(f"Credits: **{state['credits']}**")
    st.write(f"Scenarios run: **{state['scenarios_run']}**")
    st.write(f"Reports generated: **{state['reports_generated']}**")
    if st.checkbox("Show transaction log"):
        st.table(pd.DataFrame(state["transactions"]).head(10))

# -------------------------------------
# Center area: visualization & report
# -------------------------------------
st.markdown("---")
st.header("Results & Charts")

if "last_sim" in st.session_state and st.session_state["last_sim"] is not None:
    last = st.session_state["last_sim"]
    ts = pd.DataFrame(last["ts"])
    # plot projected cash
    fig = px.line(ts, x="month_index", y="projected_cash", title="Projected Cash Over Time",
                  labels={"month_index": "Month index", "projected_cash": "Projected Cash (₹)"})
    st.plotly_chart(fig, use_container_width=True)

    # quick text summary
    summary = last["summary"]
    st.subheader("Interpretation")
    if summary["monthly_net"] >= 0:
        st.success("Projected monthly net is non-negative → cash grows over time.")
    else:
        st.warning("Projected monthly net is negative → runway will be consumed.")
    st.write("Key numbers:")
    st.write(f"- Current cash: ₹{summary['current_cash']:,.2f}")
    st.write(f"- Projected cash after period: ₹{summary['projected_cash_after_months']:,.2f}")
    st.write(f"- Estimated runway: {'Growing' if summary['runway_months']==float('inf') else f'{summary['runway_months']:.2f} months'}")
    st.write(f"- New headcount (if hires applied): {summary['new_headcount']}")

    # export report
    st.markdown("### Export Report (PDF)")
    if st.button("Export PDF Report"):
        REPORT_COST = 2  # credits per report
        charge(REPORT_COST, "report", "Exported PDF report")
        # generate PDF bytes
        params = last["params"]
        ts_plot_df = pd.DataFrame(last["ts"])
        pdf_bytes = generate_pdf_report(summary, params, ts_plot_df)
        st.download_button("Download report", pdf_bytes, file_name=f"cfo_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
        st.success("Report generated and charged.")

else:
    st.info("Run a simulation to see results and generate a report.")

# -------------------------------------
# Extra: small analytics and README
# -------------------------------------
st.markdown("---")
st.header("Notes & Submission Checklist")
st.markdown("""
- This demo app simulates Flexprice billing by deducting credits per scenario and per report.
- Pathway integration is simulated: use the *Simulate Pathway update* button to mock fresh external data that affects revenue/expenses.
- To prepare for submission: include your source code, a short performance_report.pdf summarizing approach and results, and this app's screenshots or demo link.
""")

# Save state before exit
save_state()
