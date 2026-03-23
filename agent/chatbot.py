"""
Terminal chatbot for electricity forecasting
Run: python chatbot.py
"""
from __future__ import annotations
import os, warnings
import pandas as pd
warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

# from models.predict import predict_linear_regression, predict_prophet, predict_sarimax
from models.predict import predict_linear_regression, predict_prophet

# ── dataset ──────────────────────────────────────────────────────────────────
_df: pd.DataFrame | None = None

def get_df() -> pd.DataFrame:
    if _df is None:
        raise RuntimeError("Dataset not loaded.")
    return _df

# ── tools ─────────────────────────────────────────────────────────────────────

@tool
def run_linear_regression(client_id: str, horizon_hours: int) -> str:
    """Forecast electricity consumption using Linear Regression.
    Args:
        client_id: Client identifier e.g. MT_001
        horizon_hours: Number of hours to forecast e.g. 48
    """
    return predict_linear_regression(client_id, horizon_hours, get_df()).to_summary()

@tool
def run_prophet(client_id: str, horizon_hours: int) -> str:
    """Forecast electricity consumption using Prophet.
    Args:
        client_id: Client identifier e.g. MT_001
        horizon_hours: Number of hours to forecast e.g. 48
    """
    return predict_prophet(client_id, horizon_hours, get_df()).to_summary()

# @tool
# def run_sarimax(client_id: str, horizon_hours: int) -> str:
#     """Forecast electricity consumption using SARIMAX.
#     Args:
#         client_id: Client identifier e.g. MT_001
#         horizon_hours: Number of hours to forecast e.g. 48
#     """
#     return predict_sarimax(client_id, horizon_hours, get_df()).to_summary()

# ── system prompt ─────────────────────────────────────────────────────────────

SYSTEM = """You are an electricity consumption forecasting assistant.

When the user asks for a forecast:
1. Call BOTH tools (run_linear_regression, run_prophet) for EACH client mentioned.
2. Present the raw results from each tool exactly as returned, clearly labelled by model and client.
3. Do not add any analysis, commentary, comparison, or interpretation. Just show the numbers.

Rules:
- The dataset covers 2011-2014. All forecasts start from 2015-01-01.
- Client IDs look like MT_001, MT_315. If the user says 'client 1' use MT_001, 'client 315' use MT_315.
- Never skip a model. Never skip a client.
- If a tool returns an error, show it and continue with the others."""

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global _df

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set...")

    print("Loading dataset...")
    _df = pd.read_parquet(
        r"..\Datasets\processed_electricity_data.parquet",
        engine="pyarrow"
    )
    _df["Is_Weekend"] = _df["Is_Weekend"].astype(int)
    _df["Is_Holiday"] = _df["Is_Holiday"].astype(int)
    _df["Date"]       = pd.to_datetime(_df["Date"])
    _df = _df.sort_values(["ClientID","Date"]).reset_index(drop=True)
    print(f"Ready — {_df['ClientID'].nunique()} clients loaded.\n")

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)
    tools = [run_linear_regression, run_prophet]
    agent = create_react_agent(llm, tools)

    chat_history = []

    print("Electricity Forecast Chatbot  (type 'exit' to quit)")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break
        if not user_input:
            continue

        try:
            messages = [SystemMessage(content=SYSTEM)] + chat_history + [HumanMessage(content=user_input)]
            result   = agent.invoke({"messages": messages})
            reply    = result["messages"][-1].content
            print(f"\nAgent:\n{reply}")

            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=reply))

        except Exception as e:
            print(f"\n[Error] {e}")

if __name__ == "__main__":
    main()