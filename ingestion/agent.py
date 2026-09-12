"""
agent.py
Layer 4 (Agent Orchestration) - The core "agentic" loop of PortfolioPilot.
Given a user's investment research question, this:
  1. Sends the question + available tools to Claude
  2. Lets Claude decide which tool(s) to call (it may call several, or none)
  3. Executes those tool calls locally against your Layer 1-3 data
  4. Feeds the results back to Claude
  5. Claude synthesizes a final answer, grounded in the tool outputs, and
     is explicitly prompted to flag its own confidence / data limitations

This is the piece that demonstrates the "agentic" part of your project: the
model itself decides which of get_ratios / get_price_signals / search_filings
to call based on the question, rather than a hardcoded if/else pipeline.

Setup:
    pip install anthropic
    setx ANTHROPIC_API_KEY "your-key-here"      (Windows, then restart terminal)
    export ANTHROPIC_API_KEY="your-key-here"    (Mac/Linux)

Usage:
    python agent.py --query "How is Nike's financial health trending, and are there any red flags?"
"""

import argparse
import json
import os

import anthropic

from tools import TOOL_SCHEMAS, TOOL_FUNCTIONS

MODEL_NAME = "claude-sonnet-4-5"  # swap for a different Claude model if you prefer

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


def run_agent(user_query: str, max_tool_iterations: int = 5, verbose: bool = True) -> tuple[str, list]:
    """Returns (final_answer_text, tool_call_log) where tool_call_log is a
    list of {"tool": name, "input": kwargs, "output": result} dicts, in the
    order the tools were actually called."""
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment

    messages = [{"role": "user", "content": user_query}]
    tool_call_log = []

    for iteration in range(max_tool_iterations):
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        # Collect any tool_use blocks Claude wants executed this turn
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        if not tool_use_blocks:
            # No more tools needed - this is the final answer
            final_text = "".join(b.text for b in response.content if b.type == "text")
            return final_text, tool_call_log

        # Record the assistant's turn (including its tool call requests)
        messages.append({"role": "assistant", "content": response.content})

        # Execute each requested tool locally and collect results
        tool_results = []
        for block in tool_use_blocks:
            fn = TOOL_FUNCTIONS.get(block.name)
            if verbose:
                print(f"  [tool call] {block.name}({block.input})")

            if fn is None:
                result = {"error": f"Unknown tool: {block.name}"}
            else:
                try:
                    result = fn(**block.input)
                except Exception as e:
                    result = {"error": f"Tool '{block.name}' raised an exception: {e}"}

            tool_call_log.append({"tool": block.name, "input": block.input, "output": result})

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result, default=str),
            })

        # Feed tool results back so Claude can continue reasoning or answer
        messages.append({"role": "user", "content": tool_results})

    return "Reached max tool-call iterations without a final answer. Try a narrower question.", tool_call_log


def main():
    parser = argparse.ArgumentParser(description="Ask PortfolioPilot's agent an investment research question.")
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        print('Set it with: setx ANTHROPIC_API_KEY "your-key-here"   (Windows, then reopen terminal)')
        return

    print(f"Query: {args.query}\n")
    answer, tool_log = run_agent(args.query)
    print("\n=== Final Answer ===\n")
    print(answer)


if __name__ == "__main__":
    main()
