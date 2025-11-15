import os
import requests
from langchain.tools import tool
from fuzzywuzzy import process
import dotenv
from flask import Flask, jsonify, render_template, request
import json

# Load environment variables
dotenv.load_dotenv()

# API Keys
DATA_GOV_API_KEY = os.getenv("api_mandi")          # Agmarknet API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")       # Gemini Key (optional)

# Require DATA_GOV_API_KEY always
if not DATA_GOV_API_KEY:
    raise RuntimeError("❌ Missing api_mandi (DATA_GOV_API_KEY). Please add it to environment.")

USE_AGENT = bool(GOOGLE_API_KEY)

BASE_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"


# ----------------------- TOOL FUNCTION ---------------------------------
@tool
def market_price(query: str) -> str:
    """
    Fetches agricultural market prices from the data.gov.in API.
    The query format must be: 'price of tomato in maharashtra'
    Returns structured JSON.
    """
    query = query.lower()
    if " in " not in query:
        return json.dumps({"error": "Format error. Use 'price of [commodity] in [state]'"})

    parts = query.split(" in ")
    commodity = parts[0].replace("price of ", "").strip()
    state = parts[1].strip()

    params = {
        "api-key": DATA_GOV_API_KEY,
        "format": "json",
        "filters[state]": state.title(),
        "filters[commodity]": commodity.title(),
        "limit": 100
    }

    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        return json.dumps({
            "crop": commodity,
            "state": state,
            "records": data.get("records", [])
        })

    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"API request failed: {e}"})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {e}"})


# ----------------------- AGENT SETUP ---------------------------------
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate

agent_executor = None

if USE_AGENT:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            api_key=GOOGLE_API_KEY,
            temperature=0
        )

        react_prompt = PromptTemplate(
            input_variables=["input", "agent_scratchpad"],
            template="""
You are an agriculture mandi market price assistant.
Use the tool 'market_price' when needed to fetch live mandi pricing from the Indian Government dataset.
Provide short, clear answers formatted for farmers.

If the tool returns nothing, say "Unable to find data for this crop or location."

{agent_scratchpad}
User question: {input}
"""
        )

        agent = create_react_agent(
            llm=llm,
            tools=[market_price],
            prompt=react_prompt
        )

        agent_executor = AgentExecutor(
            agent=agent,
            tools=[market_price],
            verbose=True,
            handle_parsing_errors=True
        )

        print("🌟 Gemini Agent Enabled (Hybrid mode active)")

    except Exception as e:
        print(f"⚠ Agent startup failed, running tool-only mode: {e}")
        agent_executor = None


# ----------------------- FLASK SERVER ---------------------------------
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
def query():
    data = request.get_json()
    commodity = data.get("commodity")
    state = data.get("state")
    market = data.get("market")
    question = data.get("question")  # support natural language

    try:
        # If natural language agent available and question given → use agent
        if agent_executor and question:
            print("🤖 Using AI agent for natural query")
            response = agent_executor.invoke({"input": question})
            return jsonify({"type": "agent", "response": response})

        # otherwise use tool directly
        if not commodity or not state:
            return jsonify({"error": "Commodity & State are required"}), 400

        user_input = f"price of {commodity} in {state}"
        output_text = market_price.invoke(user_input)
        data = json.loads(output_text)

        if "error" in data:
            return jsonify({"error": data["error"]})

        records = data.get("records", [])

        # Market filtering
        if market:
            market_names = [rec.get("market", "") for rec in records]
            best_match, _ = process.extractOne(market, market_names)
            records = [rec for rec in records if rec.get("market") == best_match]

        if not records:
            return jsonify({"error": "No records found"}), 404

        formatted_records = [{
            "Market": rec.get("market", ""),
            "Min Price (₹)": rec.get("min_price", ""),
            "Max Price (₹)": rec.get("max_price", ""),
            "Modal Price (₹)": rec.get("modal_price", ""),
            "Date": rec.get("arrival_date", ""),
            "Commodity": rec.get("commodity", ""),
            "State": rec.get("state", "")
        } for rec in records]

        return jsonify({
            "type": "table",
            "data": {
                "crop": data.get("crop", ""),
                "state": data.get("state", ""),
                "records": formatted_records
            }
        })

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
