"""
agent_gemini.py
Layer 4 (Agent Orchestration) - Same job as agent.py, but using Google's
Gemini API instead of Claude. Reuses tools.py unchanged - only the LLM
provider differs.

Uses the current `google-genai` SDK (NOT the deprecated `google-generativeai`
package). Passing plain Python functions as tools enables this SDK's
"automatic function calling": Gemini decides which tool(s) to call based on
their docstrings/type hints, the SDK executes them, and feeds results back -
all in one generate_content() call, no manual loop needed.

Setup:
    pip uninstall genai -y          (in case the wrong PyPI package "genai" got installed)
    pip install google-genai
    Get a key from https://aistudio.google.com/apikey
    setx GOOGLE_API_KEY "your-key-here"      (Windows, then restart terminal)
    export GOOGLE_API_KEY="your-key-here"    (Mac/Linux)

Usage:
    python agent_gemini.py --query "How is Nike's financial health trending, and are there any red flags?"

Note on model name: Gemini model names change fairly often. If MODEL_NAME
below 404s, list what's currently available to your key:
    python -c "from google import genai; import os; c = genai.Client(api_key=os.environ['GOOGLE_API_KEY']); [print(m.name) for m in c.models.list()]"
"""

import argparse
import functools
import os

from google import genai
from google.genai import types

from tools import get_ratios, get_price_signals, search_filings

MODEL_NAME = "gemini-3.6-flash"  # swap if list_models() (see docstring) shows a different name

SYSTEM_PROMPT = """\
You are PortfolioPilot, an agentic research assistant for investment decisions.

You have access to tools that pull from a pre-fetched local dataset of SEC \
filings, computed financial ratios, and price/volatility signals - not live \
market data. Use whichever tools are relevant to answer the user's question; \
you may call more than one tool, and you may call the same tool for different \
tickers if the question is comparative.

When you give your final answer:
- Ground every claim in a specific tool result (cite the ratio, the filing \
  passage and its source URL, or the price signal you're relying on).
- Explicitly state your confidence level and *why* - e.g. flag if a ratio \
  is based on old/thin data, if price signals and fundamentals disagree, or \
  if you could not find a filing passage that directly answers part of the \
  question.
- Do not present this as financial advice; frame it as research synthesis \
  the user should verify before acting on.
"""


def _make_logged_tools(log: list):
    """
    Builds fresh wrapped tool functions that append to `log` when called.
    A fresh set is built per run_agent() call (not once at import time) so
    concurrent/successive calls (e.g. from a Streamlit app) don't share or
    leak entries between each other.
    """
    def wrap(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            print(f"  [tool call] {fn.__name__}({kwargs})")
            result = fn(*args, **kwargs)
            log.append({"tool": fn.__name__, "input": kwargs, "output": result})
            return result
        return wrapper

    return [wrap(get_ratios), wrap(get_price_signals), wrap(search_filings)]


def run_agent(user_query: str) -> tuple[str, list]:
    """Returns (final_answer_text, tool_call_log) where tool_call_log is a
    list of {"tool": name, "input": kwargs, "output": result} dicts, in the
    order the tools were actually called."""
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    tool_call_log = []
    tools = _make_logged_tools(tool_call_log)

    # Automatic function calling is only recommended via a chat session
    # (not a one-off generate_content call) - see the SDK's own warning.
    chat = client.chats.create(
        model=MODEL_NAME,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=tools,
        ),
    )
    response = chat.send_message(user_query)
    return response.text, tool_call_log


def main():
    parser = argparse.ArgumentParser(description="Ask PortfolioPilot's agent (Gemini version) an investment research question.")
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    if not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY environment variable is not set.")
        print('Set it with: setx GOOGLE_API_KEY "your-key-here"   (Windows, then reopen terminal)')
        return

    print(f"Query: {args.query}\n")
    answer, tool_log = run_agent(args.query)
    print("\n=== Final Answer ===\n")
    print(answer)


if __name__ == "__main__":
    main()
