"""
agent/chatbot.py
----------------
Terminal chatbot for unified electricity forecasting (LR, Prophet, SARIMAX, NST).
Handles Cluster mapping and model/mode selection via LLM tools.
"""
from __future__ import annotations
import os, warnings
import pandas as pd
from dotenv import load_dotenv
warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from inference.predict import predict_power

# DATASET
_df: pd.DataFrame | None = None

def get_df() -> pd.DataFrame:
    if _df is None:
        raise RuntimeError("Dataset not loaded.")
    return _df

# TOOLS 

@tool
def run_forecast(client_id: str, model: str = "lr", mode: str = "day_ahead", horizon_hours: int = 48) -> str:
    """Forecast electricity consumption using a specific model and operational mode.
    Args:
        client_id: Client identifier e.g. MT_001
        model: Model type ('lr', 'prophet', 'sarimax', 'nst')
        mode: Mode ('day_ahead', 'long_term')
        horizon_hours: Number of hours to forecast (default 48)
    """
    # --- AUTO-FORMAT CLIENT ID ---
    client_id = str(client_id).strip().upper()
    if client_id.isdigit():
        client_id = f"MT_{int(client_id):03d}"
    elif client_id.startswith("MT_"):
        parts = client_id.split("_")
        if len(parts) == 2 and parts[1].isdigit():
            client_id = f"MT_{int(parts[1]):03d}"
    # -----------------------------
    
    try:
        df = get_df()
        result = predict_power(client_id, model, mode, df, horizon_hours)
        return result.to_summary()
    except Exception as e:
        return f"[Error] Forecast failed for {client_id}: {str(e)}"



@tool
def get_client_info(client_id: str) -> str:
    """Retrieve historical summary, behavioral cluster, and volume category for a specific client.
    Args:
        client_id: Client identifier e.g. MT_001
    """
    # --- AUTO-FORMAT CLIENT ID ---
    client_id = str(client_id).strip().upper()
    if client_id.isdigit():
        client_id = f"MT_{int(client_id):03d}"
    elif client_id.startswith("MT_"):
        parts = client_id.split("_")
        if len(parts) == 2 and parts[1].isdigit():
            client_id = f"MT_{int(parts[1]):03d}"
    # -----------------------------
    
    try:
        df = get_df()
        df_c = df[df["ClientID"] == client_id]
        
        if df_c.empty:
            return f"[Error] No data found for client {client_id}. Ensure it is between MT_001 and MT_370."
            
        # Extract metadata
        cluster_id = int(df_c["Cluster"].iloc[0])
        volume_cat = str(df_c["Consumer_Category"].iloc[0])
        
        # Calculate historical metrics
        mean_kw = round(df_c["Consumption"].mean(), 2)
        max_kw = round(df_c["Consumption"].max(), 2)
        total_kwh = round(df_c["Consumption"].sum() * 0.25, 2) # 15-min intervals to kWh
        
        # Human-readable cluster names based on our report mapping
        cluster_map = {
            0: "Standard Daytime Business",
            1: "Standard Residential",
            2: "Extended Commercial / Mixed-Use",
            3: "Split-Shift / Siesta Profile",
            4: "Night-Shift Industrial"
        }
        behavior = cluster_map.get(cluster_id, f"Cluster {cluster_id}")
        
        return (
            f"--- CLIENT PROFILE: {client_id} ---\n"
            f"Behavioral Shape : {behavior} (Cluster {cluster_id})\n"
            f"Volume Tier      : {volume_cat}\n"
            f"Historical Mean  : {mean_kw} kW per 15-min\n"
            f"Historical Peak  : {max_kw} kW\n"
            f"Total Energy Used: {total_kwh:,.0f} kWh (2011-2014)\n"
        )
    except Exception as e:
        return f"[Error] Could not retrieve info for {client_id}: {str(e)}"

# SYSTEM PROMPT 
SYSTEM = """You are an Expert Energy Analyst AI. 

Your objective is to provide actionable electricity consumption forecasts and historical profiling for a portfolio of Portuguese clients (IDs: MT_001 to MT_370) based on historical data (2011-2014).

TOOL USAGE STRATEGY:
1. If a user asks for general information, historical data, or the profile of a client, ALWAYS invoke the `get_client_info` tool first.
2. If a user requests a forecast, ALWAYS invoke the `run_forecast` tool. Unless specified otherwise, run the tool TWICE to provide a comparative benchmark:
   - Run a baseline model (model='lr' or 'prophet').
   - Run an advanced model (model='sarimax' or 'nst').

PARAMETERS:
- `mode`: Use 'day_ahead' (horizon 24-48h) for operational spot-market queries. Use 'long_term' (horizon 720h+) for hedging/budgeting queries.
- `client_id`: Auto-correct user input (e.g., '13' -> 'MT_013').

RESPONSE FORMAT:
- Use clean Markdown lists/tables.
- Be analytical. If retrieving client info, explain what their "Behavioral Shape" means practically.
- If providing a forecast, explain *why* the models might differ and infer business insights.
"""

# Always load environment variables from .env
load_dotenv()

# ── MAIN ─

def main():
    global _df

    api_key = os.environ.get("OPENAI_KEY")
    if not api_key:
        print("[Error] OPENAI_KEY not found in environment.")
        return

    print("Loading dataset...")
    # Using relative path from agent/ directory
    parquet_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Datasets", "processed_electricity_data.parquet")
    
    try:
        _df = pd.read_parquet(parquet_path, engine="pyarrow")
        print(f"Ready — {_df['ClientID'].nunique()} clients mapped to clusters.\n")
    except Exception as e:
        print(f"[Critical Error] Could not load dataset at {parquet_path}: {e}")
        return

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)
    tools = [run_forecast, get_client_info]
    agent = create_react_agent(llm, tools)

    chat_history = []
    print("Electricity Analyst Bot  - (type 'exit' to quit)")

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