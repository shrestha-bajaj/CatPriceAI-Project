from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #07111f;
            --panel: rgba(255,255,255,0.95);
            --panel-dark: #0f172a;
            --text: #0f172a;
            --muted: #64748b;
            --accent: #2563eb;
            --accent-2: #14b8a6;
            --warn: #f59e0b;
        }
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #f8fbff 0%, #eef4ff 45%, #fdf2f8 100%);
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }
        .hero {
            padding: 1.2rem 1.4rem;
            border-radius: 18px;
            background: linear-gradient(120deg, #0f172a 0%, #1d4ed8 100%);
            color: white;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.16);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin-bottom: 0.2rem;
            font-size: 2rem;
        }
        .hero p {
            margin: 0;
            color: #dbeafe;
            font-size: 0.98rem;
        }
        .status-pill {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.16);
            font-size: 0.84rem;
            margin-top: 0.6rem;
        }
        .metric-card {
            border-radius: 16px;
            padding: 0.9rem 1rem;
            background: var(--panel);
            border: 1px solid #e2e8f0;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
            height: 100%;
        }
        .metric-title {
            color: var(--muted);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.3rem;
        }
        .metric-value {
            color: var(--text);
            font-size: 1.35rem;
            font-weight: 700;
        }
        .metric-delta {
            color: var(--accent-2);
            font-size: 0.82rem;
            margin-top: 0.2rem;
        }
        div[data-testid="stTabs"] button {
            border-radius: 999px 999px 0 0;
            padding: 0.45rem 0.8rem;
        }
        .stDataFrame {
            border-radius: 14px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_styles()


def resolve_data_path(filename: str) -> Path:
    candidates = [
        DATA_DIR / filename,
        BASE_DIR / filename,
        BASE_DIR / "ml" / filename,
        BASE_DIR / "rag" / "docs" / filename,
        BASE_DIR / "rag" / "vectorstore" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Required file not found: {filename}")


@st.cache_data(show_spinner=False)
def load_exposure_summary() -> pd.DataFrame:
    return pd.read_csv(resolve_data_path("exposure_summary.csv"))


@st.cache_data(show_spinner=False)
def load_pricing_output() -> pd.DataFrame:
    return pd.read_csv(resolve_data_path("pricing_output.csv"))


@st.cache_data(show_spinner=False)
def load_market_rol() -> pd.DataFrame:
    return pd.read_csv(resolve_data_path("market_rol.csv"))


@st.cache_data(show_spinner=False)
def load_portfolio() -> pd.DataFrame:
    return pd.read_csv(resolve_data_path("portfolio.csv"))


@st.cache_data(show_spinner=False)
def load_loss_summary() -> pd.DataFrame:
    return pd.read_csv(resolve_data_path("loss_summary.csv"))


@st.cache_data(show_spinner=False)
def load_feature_importance() -> pd.DataFrame:
    return pd.read_csv(resolve_data_path("feature_importance.csv"))


@st.cache_data(show_spinner=False)
def load_method_comparison() -> pd.DataFrame:
    return pd.read_csv(resolve_data_path("method_comparison.csv"))


@st.cache_data(show_spinner=False)
def load_burning_cost() -> pd.DataFrame:
    return pd.read_csv(resolve_data_path("burning_cost.csv"))


@st.cache_data(show_spinner=False)
def load_treaty_text() -> str:
    for filename in ["cat_xl_treaty_wording.txt", "underwriting_guidelines.txt"]:
        try:
            return resolve_data_path(filename).read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
    return ""


def require_files() -> list[str]:
    required_files = [
        "exposure_summary.csv",
        "pricing_output.csv",
        "market_rol.csv",
        "portfolio.csv",
        "loss_summary.csv",
        "feature_importance.csv",
        "method_comparison.csv",
        "burning_cost.csv",
        "cat_xl_treaty_wording.txt",
    ]
    missing = []
    for name in required_files:
        try:
            resolve_data_path(name)
        except FileNotFoundError:
            missing.append(name)
    return missing


def summarize_exposure(portfolio: pd.DataFrame, exposure_summary: pd.DataFrame) -> tuple[float, float, bool]:
    portfolio_tiv = float(portfolio["tiv_usd"].sum())
    exposure_tiv = float(exposure_summary.get("Total_TIV", exposure_summary.get("total_tiv", pd.Series([0.0]))).iloc[0])
    return portfolio_tiv, exposure_tiv, abs(portfolio_tiv - exposure_tiv) < 1_000_000


def call_openai_chat(api_key: str | None, system_prompt: str, user_prompt: str, max_tokens: int = 250) -> str:
    if not api_key:
        raise RuntimeError("No API key provided")

    try:
        import openai
    except Exception as exc:
        raise RuntimeError(f"OpenAI SDK unavailable: {exc}") from exc

    if hasattr(openai, "OpenAI"):
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return response["choices"][0]["message"]["content"].strip()


def build_agent_answer(question: str, pricing: pd.DataFrame, treaty_text: str, api_key: str | None = None) -> str:
    q = question.lower()
    pricing_row = pricing.iloc[0]
    recommendation = str(pricing_row.get("recommendation", "")).strip().upper()
    technical_rol_pct = float(pricing_row.get("technical_rol_pct", pricing_row.get("technical_rol", 0.0) * 100))
    premium = float(pricing_row.get("technical_premium_usd", pricing_row.get("technical_premium", 0.0)))

    if api_key:
        try:
            prompt = (
                "You are a pricing assistant. Use only the data available here to answer the question.\n"
                f"Question: {question}\n\n"
                f"Pricing: {pricing_row.to_dict()}\n"
                f"Treaty text snippet: {treaty_text[:1500]}"
            )
            return call_openai_chat(
                api_key,
                "You are a precise underwriting assistant.",
                prompt,
            )
        except Exception as exc:
            fallback = None
            if any(term in q for term in ["quote", "recommend", "should we"]):
                if recommendation == "QUOTE":
                    fallback = (
                        f"Yes. The current underwriting view supports a quote: recommendation is {recommendation}, "
                        f"technical ROL is {technical_rol_pct:.1f}%, and the indicative premium is ${premium:,.0f}."
                    )
                else:
                    fallback = (
                        f"No. The current numbers do not support a quote: recommendation is {recommendation}, "
                        f"technical ROL is {technical_rol_pct:.1f}%.")
            elif "premium" in q:
                fallback = f"Indicative technical premium is ${premium:,.0f}."
            elif "rol" in q:
                fallback = f"Technical ROL is {technical_rol_pct:.1f}% based on the current pricing output."
            elif "excluded" in q or "peril" in q:
                for line in treaty_text.splitlines():
                    if "EXCLUDED" in line.upper() or "EXCLUDED:" in line.upper():
                        fallback = line.strip()
                        break
            if fallback:
                return f"{fallback}\n\n(Note: OpenAI pricing error: {exc})"
            return f"OpenAI pricing error: {exc}. Using local pricing router instead."

    if any(term in q for term in ["quote", "recommend", "should we"]):
        if recommendation == "QUOTE":
            return (
                f"Yes. The current underwriting view supports a quote: recommendation is {recommendation}, "
                f"technical ROL is {technical_rol_pct:.1f}%, and the indicative premium is ${premium:,.0f}."
            )
        return (
            f"No. The current numbers do not support a quote: recommendation is {recommendation}, "
            f"technical ROL is {technical_rol_pct:.1f}%."
        )

    if "premium" in q:
        return f"Indicative technical premium is ${premium:,.0f}."

    if "rol" in q:
        return f"Technical ROL is {technical_rol_pct:.1f}% based on the current pricing output."

    if "excluded" in q or "peril" in q:
        for line in treaty_text.splitlines():
            if "EXCLUDED" in line.upper() or "EXCLUDED:" in line.upper():
                return line.strip()

    return (
        "The agent is using the local underwriting outputs and treaty wording. "
        "Ask about quoting, premium, ROL, or treaty exclusions for a more specific answer."
    )


def build_treaty_answer(question: str, treaty_text: str, api_key: str | None = None) -> str:
    q = question.lower()
    if api_key:
        try:
            return call_openai_chat(
                api_key,
                "You are a treaty document assistant. Answer using only the treaty text.",
                f"Treaty text:\n{treaty_text[:1500]}\n\nQuestion: {question}",
            )
        except Exception as exc:
            fallback = None
            if "excluded perils" in q or "excluded" in q:
                relevant = [line.strip() for line in treaty_text.splitlines() if "excluded" in line.lower()]
                if relevant:
                    fallback = "\n".join(relevant[:8])
            elif "territory" in q:
                relevant = [line.strip() for line in treaty_text.splitlines() if "territory" in line.lower()]
                if relevant:
                    fallback = "\n".join(relevant[:6])
            elif "attachment" in q or "limit" in q:
                relevant = [line.strip() for line in treaty_text.splitlines() if "attachment" in line.lower() or "limit" in line.lower()]
                if relevant:
                    fallback = "\n".join(relevant[:6])
            if fallback:
                return f"{fallback}\n\n(Note: OpenAI treaty fallback due to: {exc})"
            return f"OpenAI treaty answer error: {exc}. Using local text matching instead."

    if "excluded perils" in q or "excluded" in q:
        relevant = [line.strip() for line in treaty_text.splitlines() if "excluded" in line.lower()]
        if relevant:
            return "\n".join(relevant[:8])
    if "territory" in q:
        relevant = [line.strip() for line in treaty_text.splitlines() if "territory" in line.lower()]
        if relevant:
            return "\n".join(relevant[:6])
    if "attachment" in q or "limit" in q:
        relevant = [line.strip() for line in treaty_text.splitlines() if "attachment" in line.lower() or "limit" in line.lower()]
        if relevant:
            return "\n".join(relevant[:6])
    return "I could not find a direct match in the treaty wording. Try asking about excluded perils, territory, or attachment/limit."


def build_summary_table(
    exposure_summary: pd.DataFrame,
    pricing: pd.DataFrame,
    burning_cost: pd.DataFrame,
    market_rol: pd.DataFrame,
    portfolio: pd.DataFrame,
) -> pd.DataFrame:
    pricing_row = pricing.iloc[0]
    exposure_row = exposure_summary.iloc[0]
    burning_row = burning_cost.iloc[0]
    portfolio_tiv = float(portfolio["tiv_usd"].sum())

    market_avg = market_rol["market_rol"].mean()
    summary = pd.DataFrame(
        [{
            "Portfolio_TIV": portfolio_tiv,
            "Exposure_Summary_TIV": exposure_row.get("Total_TIV", exposure_row.get("total_tiv", 0.0)),
            "Technical_ROL_pct": pricing_row.get("technical_rol_pct", pricing_row.get("technical_rol", 0.0) * 100),
            "Technical_Premium_USD": pricing_row.get("technical_premium_usd", pricing_row.get("technical_premium", 0.0)),
            "Burning_Cost_ROL_pct": burning_row["burning_cost_rol_pct"],
            "Market_Benchmark_ROL_pct": market_avg * 100,
            "Recommendation": pricing_row["recommendation"],
            "Top_State": exposure_row.get("Top_State", exposure_row.get("highest_tiv_state", "Unknown")),
            "Top_Construction": exposure_row.get("Top_Construction", "Unknown"),
        }]
    )
    return summary


st.set_page_config(page_title="Cat XL Underwriting Workbench", page_icon="🛡️", layout="wide")

default_openai_key = os.getenv("OPENAI_API_KEY", "")
with st.sidebar:
    st.header("Settings")
    openai_api_key = st.text_input(
        "OpenAI API key",
        type="password",
        value=default_openai_key,
        help="Enter an optional OpenAI API key to enable richer agent and RAG responses.",
    )
    st.caption("Leave blank to use local rule-based routing and document retrieval only.")

st.markdown(
    f"""
    <div class="hero">
        <h1>Cat XL Underwriting Workbench</h1>
        <p>Executive-ready review of exposure, pricing, market cycles, treaty wording, and AI-led underwriting insight.</p>
        <div class="status-pill">Updated {date.today().strftime('%Y-%m-%d')}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

missing = require_files()
if missing:
    st.error("Missing required files: " + ", ".join(missing))
    st.info("Run python 01_generate_data.py from the project root to regenerate the underwriting datasets.")
    st.stop()

portfolio = load_portfolio()
exposure_summary = load_exposure_summary()
pricing = load_pricing_output()
burning_cost = load_burning_cost()
market_rol = load_market_rol()
loss_summary = load_loss_summary()
feature_importance = load_feature_importance()
method_comparison = load_method_comparison()
treaty_text = load_treaty_text()

portfolio_tiv, exposure_tiv, tiv_match = summarize_exposure(portfolio, exposure_summary)
pricing_row = pricing.iloc[0]

st.success("Data loaded successfully. The app is ready for underwriting review.")


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📊 Exposure", "💰 Pricing", "📈 Market", "🤖 AI Agent", "📄 Treaty Q&A", "🧾 Summary"]
)

with tab1:
    st.subheader("Exposure Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-card'><div class='metric-title'>Portfolio TIV</div><div class='metric-value'>${:,.0f}</div><div class='metric-delta'>Synthetic cedant portfolio</div></div>".format(portfolio_tiv), unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'><div class='metric-title'>Exposure Summary TIV</div><div class='metric-value'>${:,.0f}</div><div class='metric-delta'>Reported exposure base</div></div>".format(exposure_tiv), unsafe_allow_html=True)
    with col3:
        match_label = "Aligned" if tiv_match else "Difference detected"
        st.markdown("<div class='metric-card'><div class='metric-title'>Consistency</div><div class='metric-value'>{}</div><div class='metric-delta'>{}</div></div>".format("Match" if tiv_match else "Check", match_label), unsafe_allow_html=True)

    state_tiv = portfolio.groupby("state")["tiv_usd"].sum().reset_index().sort_values("tiv_usd", ascending=False)
    fig_state = px.bar(
        state_tiv,
        x="state",
        y="tiv_usd",
        color="state",
        title="TIV by State",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig_state.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_state, width='stretch')

    st.caption(
        f"Top state: {exposure_summary.iloc[0].get('Top_State', exposure_summary.iloc[0].get('highest_tiv_state', 'Unknown'))} | "
        f"Top construction: {exposure_summary.iloc[0].get('Top_Construction', 'Unknown')}"
    )

with tab2:
    st.subheader("Pricing Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-card'><div class='metric-title'>Technical ROL</div><div class='metric-value'>{:.1f}%</div><div class='metric-delta'>Current pricing output</div></div>".format(float(pricing_row['technical_rol_pct'])), unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'><div class='metric-title'>Technical Premium</div><div class='metric-value'>${:,.0f}</div><div class='metric-delta'>Indicative premium</div></div>".format(float(pricing_row['technical_premium_usd'])), unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'><div class='metric-title'>Recommendation</div><div class='metric-value'>{}</div><div class='metric-delta'>Underwriter view</div></div>".format(str(pricing_row["recommendation"]).upper()), unsafe_allow_html=True)

    fig = go.Figure(go.Bar(
        x=["Burning Cost", "Frequency/Severity", "Exposure Rating", "Blended Pure", "Technical"],
        y=[float(pricing_row["burning_cost_rol"] * 100), float(pricing_row["freq_severity_rol"] * 100), float(pricing_row["exposure_rating_rol"] * 100), float(pricing_row["blended_pure_rol"] * 100), float(pricing_row["technical_rol"] * 100)],
        marker_color=["#60a5fa", "#2563eb", "#1d4ed8", "#f59e0b", "#ef4444"],
        text=[f"{v:.1f}%" for v in [float(pricing_row["burning_cost_rol"] * 100), float(pricing_row["freq_severity_rol"] * 100), float(pricing_row["exposure_rating_rol"] * 100), float(pricing_row["blended_pure_rol"] * 100), float(pricing_row["technical_rol"] * 100)]],
        textposition="outside",
    ))
    fig.update_layout(title="ROL Build", yaxis_title="ROL %", template="plotly_white", margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width='stretch')

    st.caption(f"Technical ROL in pricing_output.csv: {round(float(pricing_row['technical_rol_pct']), 1)} | Burning cost ROL: {round(float(burning_cost.iloc[0]['burning_cost_rol_pct']), 2)}")

with tab3:
    st.subheader("Market Cycle")
    market_yearly = (
        market_rol.groupby("year")["market_rol"].mean().reset_index().sort_values("year")
    )
    fig_market = px.line(
        market_yearly,
        x="year",
        y="market_rol",
        markers=True,
        title="Market ROL by Year",
        line_shape="spline",
        color_discrete_sequence=["#2563eb"],
    )
    fig_market.add_vrect(x0=2005.5, x1=2008.5, annotation_text="Post-Katrina spike", annotation_position="top left", fillcolor="#fda4af", opacity=0.25)
    fig_market.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_market, width='stretch')
    st.caption("Hover over the chart to identify the hardening period after Katrina and the later Ian spike.")

with tab4:
    st.subheader("AI Agent")
    st.write("Use the agent for quick underwriting guidance based on current pricing and treaty terms.")
    question = st.text_area("Ask the underwriting agent", value="Should we quote this treaty?")
    if st.button("Ask Agent"):
        if openai_api_key:
            st.info("Using OpenAI to generate the answer.")
        answer = build_agent_answer(question, pricing, treaty_text, api_key=openai_api_key)
        st.success(answer)
    else:
        st.info("Ask about quotation, premium, ROL, or treaty exclusions.")

with tab5:
    st.subheader("Treaty Q&A")
    treaty_question = st.text_input("Search the treaty wording", value="What are the excluded perils?")
    if st.button("Search Treaty"):
        if openai_api_key:
            st.info("Using OpenAI to generate a treaty-aware answer.")
        answer = build_treaty_answer(treaty_question, treaty_text, api_key=openai_api_key)
        st.info(answer)
    else:
        st.info("Try questions about excluded perils, territory, attachment, or limit.")

with tab6:
    st.subheader("Underwriting Summary")
    summary_df = build_summary_table(exposure_summary, pricing, burning_cost, market_rol, portfolio)
    st.dataframe(summary_df, hide_index=True)

    csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Export Memo",
        data=csv_bytes,
        file_name="pricing_memo.csv",
        mime="text/csv",
        width='stretch',
    )
