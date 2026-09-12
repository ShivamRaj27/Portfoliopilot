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
This opens a local browser tab (usually http://localhost:8501) - it is not
a public website, just a local web server on your own machine.

API keys: enter them in the sidebar each session (kept in memory only, never
written to disk) - simpler for a group project than relying on setx/export
persisting correctly across everyone's machines.
"""

from pathlib import Path

import streamlit as st

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(page_title="PortfolioPilot", page_icon="📊", layout="wide")


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
    st.title("📊 PortfolioPilot")
    st.caption("An agentic research assistant for investment decisions — grounded in SEC filings, computed ratios, and price signals.")

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
    query = st.text_area(
        "Your question",
        placeholder=f"e.g. How is {ticker_hint}'s financial health trending, and are there any red flags?",
        height=100,
        label_visibility="collapsed",
    )

    st.caption("Or try an example:")
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
        st.markdown(answer)

        st.subheader("How the agent got there")
        render_tool_call_log(tool_call_log)

        st.caption(
            "This is research synthesis grounded in pre-fetched SEC filings, computed ratios, "
            "and price signals — not live market data or financial advice. Verify before acting on it."
        )


if __name__ == "__main__":
    main()
