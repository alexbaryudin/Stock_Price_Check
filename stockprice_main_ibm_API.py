# This version of the script is creating API endpoint to request a stock price for a symbol.
# In order to run it, uvicorn server needs to be started: uvicorn stockprice_main_ibm_API:app --reload
# and then http://127.0.0.1:8000/docs# url will open documentation page where it could be tested


import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.agents import Tool, AgentType, initialize_agent
from langchain_openai import ChatOpenAI
import yfinance as yf
from langchain_ibm import WatsonxLLM




load_dotenv()

# Set up FastAPI
app = FastAPI()

# Load credentials from environment
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": os.environ['WATSONX_API_KEY'],
    "project_id": os.environ['WATSON_ML_PROJECT']
}

model_param = {
    "decoding_method": "greedy",
    "temperature": 0,
    "min_new_tokens": 5,
    "max_new_tokens": 500
}

# Define a function to get stock price using yfinance
def get_stock_price(symbol: str) -> str:
    try:
        ticker = yf.Ticker(symbol)
        stock_price = ticker.history(period="1d")["Close"].iloc[-1]
        return f"The current price of {symbol.upper()} is ${stock_price:.2f}"
    except Exception as e:
        return f"Error fetching stock price: {str(e)}"

# Create a Tool with the get_stock_price function
stock_price_tool = Tool(
    name="Stock Price Checker",
    func=get_stock_price,
    description="Fetches the latest stock price for a given stock symbol (e.g., AAPL for Apple Inc.)"
)

# Set up the LLM with IBM Watsonx
model_id = "meta-llama/llama-3-2-90b-vision-instruct"

llm = WatsonxLLM(
    model_id=model_id,
    url=credentials.get("url"),
    apikey=credentials.get("apikey"),
    project_id=credentials.get("project_id"),
    params=model_param
)

# Create the agent
agent = initialize_agent(
    tools=[stock_price_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Define the input model for the API
class StockRequest(BaseModel):
    symbol: str



# Define the API endpoint for getting stock price
@app.post("/get-stock-price")
async def get_stock_price_endpoint(request: StockRequest):
    try:
        response = agent.run(f"What is the current stock price of {request.symbol}?")
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
