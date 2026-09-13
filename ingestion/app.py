"""
app.py
Layer 5 (Interface) - A Streamlit UI for PortfolioPilot. Wraps the agent
loop built in Layer 4 (agent.py / agent_gemini.py) so a user can ask an
investment research question in a browser instead of the command line, and
see which tools the agent chose to call and why (the "agentic" behavior),
not just the final answer.

Also includes a dedicated "ML Model Lab" tab that calls the trained models
(classifiers, Lasso, K-Means, Isolation Forest) DIRECTLY, independent of
whether the chat agent happens to invoke them for a given question - this
guarantees the real, trained ML output is visible and demonstrable, not
just implied by an LLM's prose.

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

import pandas as pd
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


def _classifier_plain_summary(result: dict) -> str:
    """Builds a deterministic, plain-language explanation of predict_direction()'s
    output - no LLM call needed, so this works even without an API key."""
    recommended = result.get("recommended")
    if not recommended:
        return "No summary available - a best model couldn't be determined."

    ticker = result["ticker"]
    direction = recommended["predicted_direction"]
    prob = recommended["probability_up"]
    model = recommended["model"]
    accuracy = recommended["test_accuracy"]

    diff = abs(prob - 0.5)
    if diff < 0.05:
        lean = "essentially a toss-up, with only the slightest lean"
    elif diff < 0.15:
        lean = "a mild lean"
    elif diff < 0.25:
        lean = "a moderate lean"
    else:
        lean = "a fairly strong lean"

    return (
        f"Our best-performing model ({model}) shows {lean} toward **{ticker} going "
        f"{direction.upper()}** tomorrow ({prob:.0%} probability of an up move). But this "
        f"model is only right about {accuracy:.0%} of the time historically — barely better "
        f"than a coin flip — so this is a minor data point to keep in mind, not something to "
        f"act on by itself."
    )


def _lasso_plain_summary(result: dict) -> str:
    """Plain-language explanation of get_lasso_findings()'s output."""
    ticker = result["ticker"]
    kept = result.get("features_kept", [])

    if not kept:
        return (
            f"None of the 14 signals we track (recent price trends, volatility, RSI, profit "
            f"margins, debt levels, and more) reliably explain how BIG {ticker}'s next price "
            f"move will be. In plain terms: there's no dependable formula among these numbers "
            f"for predicting tomorrow's move size — a common, honest finding for actively-traded, "
            f"closely-watched stocks."
        )

    top_feature = kept[0]["feature"]
    return (
        f"{len(kept)} of the signals we track showed a strong enough relationship with next-day "
        f"price movement size to survive the model's filtering, led by **{top_feature}**. This "
        f"suggests it may carry some real predictive value — though treat this as a mild hint "
        f"rather than a rule, since overall prediction accuracy for move size remains low."
    )


def _cluster_plain_summary(result: dict) -> str:
    """Plain-language explanation of get_cluster()'s output, translating the raw
    ratio numbers into descriptive labels a non-technical reader can follow."""
    ticker = result["ticker"]
    profile = result.get("cluster_average_profile", {})
    members = [m for m in result.get("cluster_members", []) if m != ticker]

    net_margin = profile.get("net_margin")
    debt_to_equity = profile.get("debt_to_equity")
    growth = profile.get("revenue_yoy_growth")

    margin_word = ("high-margin" if net_margin is not None and net_margin > 0.25
                   else "moderate-margin" if net_margin is not None and net_margin > 0.12
                   else "lower-margin")
    leverage_word = ("low-debt" if debt_to_equity is not None and debt_to_equity < 1.5
                     else "moderate-debt" if debt_to_equity is not None and debt_to_equity < 3
                     else "high-debt")
    growth_word = ("fast-growing" if growth is not None and growth > 0.10
                   else "steadily-growing" if growth is not None and growth > 0
                   else "slow or declining")

    peer_text = f"grouped with {', '.join(members)}" if members else "not grouped with any other company in this dataset"
    return (
        f"{ticker} is {peer_text} because they share a similar overall financial profile: "
        f"**{margin_word}, {leverage_word}, and {growth_word}** revenue. This grouping is based "
        f"purely on the actual numbers (profitability, leverage, efficiency, growth) — not "
        f"official industry or sector labels."
    )


def _anomaly_plain_summary(result: dict) -> str:
    """Plain-language explanation of get_price_signals()'s two anomaly methods."""
    total_days = result.get("total_days_analyzed", 0)
    total_z = result.get("total_zscore_anomalies", 0)
    total_ml = result.get("total_ml_anomalies", 0)
    overlap = result.get("total_overlap_anomalies", 0)

    return (
        f"Over the {total_days} trading days analyzed, the simple rule-based method flagged "
        f"{total_z} unusual days, and the AI-based method flagged {total_ml}. **{overlap} day(s) "
        f"were flagged by both methods** — those are the strongest candidates for genuinely "
        f"unusual market events, since two independent approaches agree. Days flagged by only "
        f"one method are weaker signals and may just reflect that method's own blind spots."
    )


def render_classifier_results(result: dict):
    """Renders predict_direction() output as a single recommended call, backed by
    whichever trained model actually scored best on accuracy + precision - not
    three possibly-disagreeing model cards."""
    if "error" in result:
        st.error(result["error"])
        return

    st.caption(f"As of {result['as_of_date']} — predicting tomorrow's direction")

    recommended = result.get("recommended")
    if recommended:
        direction = recommended["predicted_direction"]
        st.metric(
            f"Recommended call — {recommended['model']}",
            direction.upper(),
            f"{recommended['probability_up']:.1%} probability up",
            delta_color="normal" if direction == "up" else "inverse",
        )
        st.success(f"📝 **What this means:** {_classifier_plain_summary(result)}")
    else:
        st.warning("No saved test-set results found to pick a best model - run train_classifiers.py first.")

    with st.expander("See all three models' predictions (technical detail)"):
        labels = {
            "random_forest": "Random Forest",
            "xgboost": "XGBoost",
            "logistic_regression": "Logistic Regression (baseline)",
        }
        cols = st.columns(3)
        for col, (key, label) in zip(cols, labels.items()):
            pred = result["all_model_predictions"][key]
            direction = pred["predicted_direction"]
            prob = pred["probability_up"]
            col.metric(label, direction.upper(), f"{prob:.1%} probability up",
                       delta_color="normal" if direction == "up" else "inverse")

    st.info(f"⚠️ {result['caveat']}")


def render_lasso_results(result: dict):
    """Renders get_lasso_findings() output: a plain-language summary first, then
    Lasso's official selection, then a clearly-separated raw-correlation
    diagnostic view so there's always something concrete visible even when
    Lasso's honest answer is 'none kept'."""
    if "error" in result:
        st.error(result["error"])
        return

    st.success(f"📝 **What this means:** {_lasso_plain_summary(result)}")

    kept = result["features_kept"]
    dropped = result["features_dropped"]

    with st.expander("See Lasso's technical result"):
        st.write("##### Lasso's official result")
        if kept:
            st.write("**Features Lasso kept (non-zero coefficient):**")
            st.dataframe(pd.DataFrame(kept), use_container_width=True, hide_index=True)
        else:
            st.write("**Features Lasso kept:** none")
        st.write(f"**Features Lasso dropped to exactly zero ({len(dropped)}):**")
        st.write(", ".join(dropped) if dropped else "—")
        st.caption(result["note"])

        ranking = result.get("raw_correlation_ranking")
        if ranking:
            st.divider()
            st.write("##### Diagnostic view (not a Lasso result)")
            ranking_df = pd.DataFrame(ranking)
            st.bar_chart(
                ranking_df.set_index("feature")["correlation_with_next_day_return"],
                use_container_width=True,
            )
            st.dataframe(
                ranking_df[["feature", "correlation_with_next_day_return"]],
                use_container_width=True,
                hide_index=True,
            )
            st.caption(f"⚠️ {result['raw_correlation_disclaimer']}")


def render_cluster_results(result: dict):
    """Renders get_cluster() output: plain-language summary first, then the
    technical membership + profile table."""
    if "error" in result:
        st.error(result["error"])
        return

    st.success(f"📝 **What this means:** {_cluster_plain_summary(result)}")

    with st.expander("See technical detail"):
        st.metric("Cluster", f"#{result['cluster_id']}")
        st.write(f"**Peers in this cluster:** {', '.join(result['cluster_members'])}")
        st.write("**Cluster average financial profile:**")
        profile_df = pd.DataFrame([result["cluster_average_profile"]])
        st.dataframe(profile_df, use_container_width=True, hide_index=True)
        if result.get("caveat"):
            st.info(f"⚠️ {result['caveat']}")


def render_anomaly_results(result: dict):
    """Renders get_price_signals() output: plain-language summary first, then
    both anomaly-detection methods' recent flagged days side by side."""
    if "error" in result:
        st.error(result["error"])
        return

    st.success(f"📝 **What this means:** {_anomaly_plain_summary(result)}")

    with st.expander("See recent flagged days (technical detail)"):
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Rule-based (z-score) anomalies:**")
            zscore_rows = result.get("recent_anomalies_zscore_method", [])
            if zscore_rows:
                st.dataframe(pd.DataFrame(zscore_rows), use_container_width=True, hide_index=True)
            else:
                st.caption("None flagged in the recent window.")
        with c2:
            st.write("**Isolation Forest (ML) anomalies:**")
            ml_rows = result.get("recent_anomalies_ml_method", [])
            if ml_rows:
                st.dataframe(pd.DataFrame(ml_rows), use_container_width=True, hide_index=True)
            else:
                st.caption("None flagged in the recent window.")
        st.caption(
            "Two independent anomaly-detection methods, run side by side: a simple statistical "
            "rule (|z-score| threshold) and a trained Isolation Forest model on daily return, "
            "volatility, and RSI."
        )


def render_ml_lab(ticker: str):
    """
    Dedicated panel that calls the trained ML tools DIRECTLY for the
    selected ticker - independent of the chat agent above. Guarantees real,
    trained-model output is visible on demand, regardless of what question
    (if any) the agent was asked. Each tab leads with a plain-language
    explanation of what the model found, aimed at someone without an ML
    background, followed by the technical detail in a collapsed expander.
    """
    from tools import get_cluster, get_lasso_findings, get_price_signals, predict_direction

    st.subheader(f"🔬 ML Model Lab — {ticker}")
    st.caption(
        "These run the actual trained models directly (not through the LLM agent) so you can "
        "see real, reproducible ML output on demand. Each tab answers a different question:"
    )
    st.markdown(
        "- **Direction Classifiers** → Will the stock likely go up or down tomorrow?\n"
        "- **Lasso Feature Selection** → Do any of our data signals reliably explain *how big* tomorrow's move will be?\n"
        "- **Peer Clustering** → Which other companies does this one financially resemble?\n"
        "- **Anomaly Detection** → Were there any unusual trading days recently?"
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Direction Classifiers", "🧮 Lasso Feature Selection", "🗂️ Peer Clustering", "🚨 Anomaly Detection",
    ])

    with tab1:
        st.caption("Random Forest, XGBoost, and Logistic Regression trained on lagged returns, "
                   "technical indicators, and fundamentals to predict tomorrow's direction.")
        with st.spinner("Running trained classifiers..."):
            render_classifier_results(predict_direction(ticker))

    with tab2:
        st.caption("LassoCV (5-fold cross-validated) predicting next-day return magnitude - "
                   "shows which features carry real linear signal vs. none.")
        with st.spinner("Running Lasso regression..."):
            render_lasso_results(get_lasso_findings(ticker))

    with tab3:
        st.caption("K-Means clustering on financial ratio profiles (margins, leverage, efficiency, "
                   "growth) - groups companies by their actual numbers, not sector labels.")
        with st.spinner("Loading cluster assignment..."):
            render_cluster_results(get_cluster(ticker))

    with tab4:
        st.caption("Two anomaly-detection methods on the same price history, run side by side.")
        with st.spinner("Loading price signals..."):
            render_anomaly_results(get_price_signals(ticker))


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

    main_tab, lab_tab = st.tabs(["💬 Ask a Question", "🔬 ML Model Lab"])

    with main_tab:
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

    with lab_tab:
        if not available_tickers:
            st.warning("No ticker data found in data/. Run Layers 1-3's scripts first.")
        else:
            lab_ticker = st.selectbox("Select a ticker", available_tickers, key="lab_ticker")
            render_ml_lab(lab_ticker)


if __name__ == "__main__":
    main()
