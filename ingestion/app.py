"""
app.py
Layer 5 (Interface) - A Streamlit UI for PortfolioPilot. Wraps the agent
loop built in Layer 4 (agent.py / agent_gemini.py) so a user can ask an
investment research question in a browser instead of the command line, and
see which tools the agent chose to call and why (the "agentic" behavior),
not just the final answer.

Setup:
    pip install streamlit
    (plus whichever of `anthropic` / `google-genai` you're using - see
     agent.py / agent_gemini.py for provider-specific setup)

Usage:
    streamlit run app.py

"""

from pathlib import Path

import streamlit as st

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(page_title="PortfolioPilot", page_icon=" 📊 ", layout="wide")

st.markdown("""
<style>

/* ---------- GLOBAL ---------- */

.stApp {
    background: #0B1020;
    color: #F8FAFC;
}

.main .block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* ---------- SIDEBAR ---------- */

section[data-testid="stSidebar"] {
    background: #080D1A;
    border-right: 1px solid #1E293B;
}

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #F8FAFC;
}

/* ---------- HEADINGS ---------- */

h1 {
    font-size: 42px !important;
    font-weight: 700 !important;
    letter-spacing: -1px;
}

h2 {
    font-size: 25px !important;
    margin-top: 2rem !important;
}

h3 {
    font-size: 19px !important;
}

/* ---------- TEXT ---------- */

p {
    color: #94A3B8;
}

/* ---------- INPUT ---------- */

textarea {
    background: #111827 !important;
    color: #F8FAFC !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
}

textarea:focus {
    border: 1px solid #6366F1 !important;
}

/* ---------- BUTTON ---------- */

.stButton > button {
    border-radius: 10px;
    border: 1px solid #334155;
    background: #111827;
    color: #E2E8F0;
    font-weight: 500;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: #6366F1;
    color: white;
    background: #172033;
}

/* Primary button */

button[kind="primary"] {
    background: #4F46E5 !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
}

button[kind="primary"]:hover {
    background: #6366F1 !important;
}

/* ---------- CARDS ---------- */

.pp-card {
    background: linear-gradient(
        145deg,
        #111827,
        #0F172A
    );
    border: 1px solid #1E293B;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 18px;
}

.pp-card:hover {
    border-color: #334155;
}

/* ---------- BADGES ---------- */

.pp-badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: #172554;
    color: #93C5FD;
    font-size: 12px;
    font-weight: 600;
    margin-right: 6px;
}

/* ---------- DIVIDER ---------- */

hr {
    border-color: #1E293B !important;
}

/* ---------- EXPANDERS ---------- */

details {
    background: #0F172A !important;
    border: 1px solid #1E293B !important;
    border-radius: 10px !important;
}

/* ---------- SPINNER ---------- */

.stSpinner > div {
    border-top-color: #6366F1 !important;
}

</style>
""", unsafe_allow_html=True)


def list_available_tickers() -> list:
    """Scans data/ for any ticker that has computed ratios - i.e. Layers 1-2
    have actually been run for it, so the agent has something to work with."""
    if not DATA_DIR.exists():
        return []
    return sorted({p.name.replace("_ratios.parquet", "") for p in DATA_DIR.glob("*_ratios.parquet")})


EXAMPLE_QUERIES = [
    "How is {ticker}'s financial health trending, and are there any red flags?",
    "What does {ticker} say about its main business risks in recent filings?",
    "Is {ticker}'s current stock trend supported by its underlying fundamentals?",
]


def render_tool_call_log(tool_call_log: list):
    """Renders the agent's tool calls in a readable, non-overwhelming way."""
    if not tool_call_log:
        st.caption("The agent answered directly without calling any tools.")
        return

    for i, call in enumerate(tool_call_log, start=1):
        with st.expander(f"{i}. `{call['tool']}`  —  input: {call['input']}"):
            output = call["output"]
            if isinstance(output, dict) and "error" in output:
                st.error(output["error"])
            else:
                st.json(output)


def main():
    
    st.markdown("""
<div style="margin-bottom: 30px;">

<div style="
    font-size: 14px;
    font-weight: 600;
    color: #818CF8;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 10px;
">
AI-POWERED INVESTMENT RESEARCH
</div>

<div style="
    font-size: 46px;
    font-weight: 750;
    letter-spacing: -2px;
    color: #F8FAFC;
">
PortfolioPilot
</div>

<div style="
    font-size: 18px;
    color: #94A3B8;
    max-width: 750px;
    line-height: 1.6;
">
Research companies using SEC filings, financial ratios and price signals —
with an AI agent that decides which evidence to analyze.
</div>

</div>
""", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-bottom: 25px;">
    
    <span class="pp-badge">SEC FILINGS</span>
    <span class="pp-badge">FINANCIAL RATIOS</span>
    <span class="pp-badge">PRICE SIGNALS</span>
    <span class="pp-badge">AI AGENT</span>
    
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("Setup")
        provider = st.radio("LLM provider", ["Gemini", "Claude"], horizontal=True)

        if provider == "Gemini":
            api_key_input = st.text_input("Google API key", type="password", help="Get one at aistudio.google.com/apikey")
        else:
            api_key_input = st.text_input("Anthropic API key", type="password", help="Get one at console.anthropic.com")

        st.divider()
        available_tickers = list_available_tickers()
        if available_tickers:
            st.caption(f"Data available for: {', '.join(available_tickers)}")
        else:
            st.warning("No ticker data found in data/. Run Layers 1-3's scripts first (fetch_prices.py, fetch_filings.py, compute_ratios.py, price_signals.py).")

    ticker_hint = available_tickers[0] if available_tickers else "NKE"

    st.subheader("Ask a question")
    st.markdown("""
<div class="pp-card">

<div style="
    font-size: 20px;
    font-weight: 650;
    color: #F8FAFC;
    margin-bottom: 6px;
">
Research Console
</div>

<div style="
    font-size: 14px;
    color: #64748B;
    margin-bottom: 18px;
">
Ask PortfolioPilot a company-specific investment research question.
</div>

</div>
""", unsafe_allow_html=True)
    query = st.text_area(
        "Your question",
        placeholder=f"e.g. How is {ticker_hint}'s financial health trending, and are there any red flags?",
        height=100,
        label_visibility="collapsed",
    )

    st.caption("Or try an example:")
    st.markdown("""
<div style="
    color:#64748B;
    font-size:13px;
    margin-top:12px;
    margin-bottom:8px;
">
SUGGESTED RESEARCH QUESTIONS
</div>
""", unsafe_allow_html=True)
    cols = st.columns(len(EXAMPLE_QUERIES))
    for col, template in zip(cols, EXAMPLE_QUERIES):
        example = template.format(ticker=ticker_hint)
        if col.button(example, use_container_width=True):
            query = example
            st.session_state["query_override"] = example

    if "query_override" in st.session_state:
        query = st.session_state.pop("query_override")

    ask_clicked = st.button("Ask PortfolioPilot", type="primary", disabled=not query)

    if ask_clicked:
        if not api_key_input:
            st.error(f"Enter your {provider} API key in the sidebar first.")
            return
        if not available_tickers:
            st.error("No data found - run the Layer 1-3 scripts before asking questions.")
            return

        import os

        with st.spinner("Thinking — deciding which tools to call, then synthesizing an answer..."):
            try:
                if provider == "Gemini":
                    os.environ["GOOGLE_API_KEY"] = api_key_input
                    from agent_gemini import run_agent
                else:
                    os.environ["ANTHROPIC_API_KEY"] = api_key_input
                    from agent import run_agent

                answer, tool_call_log = run_agent(query)
            except Exception as e:
                st.error(f"Something went wrong calling the agent: {e}")
                return

        st.subheader("Answer")
        st.markdown("""
<div style="
    font-size: 13px;
    font-weight: 600;
    color: #818CF8;
    letter-spacing: 1.5px;
    margin-top: 35px;
    margin-bottom: 8px;
">
RESEARCH OUTPUT
</div>

<div style="
    font-size: 26px;
    font-weight: 650;
    color: #F8FAFC;
    margin-bottom: 18px;
">
Investment Analysis
</div>
""", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="pp-card">
            {answer}
            </div>
            """,
            unsafe_allow_html=True
        )
        

        st.subheader("How the agent got there")
        render_tool_call_log(tool_call_log)

        st.caption(
            "This is research synthesis grounded in pre-fetched SEC filings, computed ratios, "
            "and price signals — not live market data or financial advice. Verify before acting on it."
        )


if __name__ == "__main__":
    main()
