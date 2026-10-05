from pathlib import Path
import importlib.util
import sqlite3

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# -------------------------------
# PROJECT PATHS
# -------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

DB_PATH = DATA_DIR / "catprice.db"
EXPOSURE_PATH = DATA_DIR / "exposure_summary.csv"
PRICING_PATH = DATA_DIR / "pricing_output.csv"
BURNING_COST_PATH = DATA_DIR / "burning_cost.csv"
ML_PRICING_PATH = DATA_DIR / "ml_pricing.csv"
METHOD_COMPARISON_PATH = DATA_DIR / "method_comparison.csv"
MARKET_ROL_PATH = DATA_DIR / "market_rol.csv"
LOSS_SUMMARY_PATH = DATA_DIR / "loss_summary.csv"

AGENT_PATH = PROJECT_ROOT / "agent" / "08_pricing_agent.py"
RAG_PATH = PROJECT_ROOT / "rag" / "07_rag_engine.py"


# -------------------------------
# FASTAPI APP
# -------------------------------
app = FastAPI(
    title="CatPriceAI Practice API",
    description="FastAPI backend for CatPriceAI pricing workbench",
    version="1.0.0"
)


# -------------------------------
# REQUEST MODEL
# -------------------------------
class QuestionRequest(BaseModel):
    question: str


# -------------------------------
# UTILITY FUNCTIONS
# -------------------------------
def load_csv(path):
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def load_single_row(path):
    records = load_csv(path)
    if len(records) == 0:
        raise HTTPException(status_code=500, detail=f"No rows in {path}")
    return records[0]


def import_module_from_file(module_name, file_path):
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"{file_path} not found")

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -------------------------------
# ROOT
# -------------------------------
@app.get("/")
def root():
    return {
        "message": "CatPriceAI API is running",
        "docs": "http://127.0.0.1:8000/docs"
    }


# -------------------------------
# HEALTH CHECK
# -------------------------------
@app.get("/health")
def health():
    files = {
        "catprice_db": DB_PATH.exists(),
        "exposure": EXPOSURE_PATH.exists(),
        "pricing": PRICING_PATH.exists(),
        "burning_cost": BURNING_COST_PATH.exists(),
        "ml_pricing": ML_PRICING_PATH.exists(),
        "method_comparison": METHOD_COMPARISON_PATH.exists(),
        "market_rol": MARKET_ROL_PATH.exists(),
        "loss_summary": LOSS_SUMMARY_PATH.exists(),
        "agent": AGENT_PATH.exists(),
        "rag": RAG_PATH.exists(),
    }

    return {
        "status": "ok" if all(files.values()) else "missing_files",
        "files": files
    }


# -------------------------------
# DATA ENDPOINTS
# -------------------------------
@app.get("/exposure")
def exposure():
    return load_single_row(EXPOSURE_PATH)


@app.get("/pricing")
def pricing():
    return load_single_row(PRICING_PATH)


@app.get("/burning-cost")
def burning_cost():
    return load_single_row(BURNING_COST_PATH)


@app.get("/ml-pricing")
def ml_pricing():
    return load_single_row(ML_PRICING_PATH)


@app.get("/method-comparison")
def method_comparison():
    return {"methods": load_csv(METHOD_COMPARISON_PATH)}


@app.get("/market-cycle")
def market_cycle():
    return {"market_cycle": load_csv(MARKET_ROL_PATH)}


@app.get("/loss-summary")
def loss_summary():
    return {"loss_summary": load_csv(LOSS_SUMMARY_PATH)}


# -------------------------------
# DATABASE QUERIES
# -------------------------------
@app.get("/portfolio/state-summary")
def portfolio_state_summary():
    if not DB_PATH.exists():
        raise HTTPException(status_code=404, detail="Database not found")

    query = """
    SELECT
        state,
        COUNT(*) AS location_count,
        ROUND(SUM(tiv), 2) AS total_tiv,
        ROUND(SUM(tiv) * 100.0 /
        (SELECT SUM(tiv) FROM portfolio), 2) AS tiv_share_pct
    FROM portfolio
    GROUP BY state
    ORDER BY total_tiv DESC
    """

    with sqlite3.connect(DB_PATH) as connection:
        df = pd.read_sql_query(query, connection)

    return {"state_summary": df.to_dict(orient="records")}


@app.get("/top-losses")
def top_losses(limit: int = 5):
    if not DB_PATH.exists():
        raise HTTPException(status_code=404, detail="Database not found")

    if limit < 1:
        raise HTTPException(status_code=400, detail="limit must be >= 1")

    if limit > 25:
        raise HTTPException(status_code=400, detail="limit max is 25")

    query = """
    SELECT
        event_year,
        event_name,
        peril,
        affected_state,
        gross_loss,
        loss_in_layer
    FROM losses
    ORDER BY gross_loss DESC
    LIMIT ?
    """

    with sqlite3.connect(DB_PATH) as connection:
        df = pd.read_sql_query(query, connection, params=(limit,))

    return {
        "limit": limit,
        "top_losses": df.to_dict(orient="records")
    }


# -------------------------------
# AI ENDPOINTS
# -------------------------------
@app.post("/treaty/ask")
def treaty_ask(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Empty question")

    try:
        rag_module = import_module_from_file("rag_engine", RAG_PATH)
        rag = rag_module.RagEngine()
        rag.load_vectorstore()

        response = rag.answer_question(question)

        return {
            "question": question,
            "answer": response["answer"],
            "sources": response["sources"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pricing/query")
def pricing_query(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Empty question")

    try:
        agent_module = import_module_from_file("pricing_agent", AGENT_PATH)
        agent_cls = getattr(agent_module, "PricingAgent", None)
        if agent_cls is None:
            raise AttributeError("PricingAgent class not available")

        agent = agent_cls()
        answer = agent.answer(question)

        return {
            "question": question,
            "answer": answer
        }

    except Exception as e:
        pricing = load_single_row(PRICING_PATH)
        q = question.lower()
        recommendation = str(pricing.get("recommendation", "")).strip().upper()
        technical_rol_pct = float(pricing.get("technical_rol_pct", pricing.get("technical_rol", 0.0) * 100))
        premium = float(pricing.get("technical_premium_usd", pricing.get("technical_premium", 0.0)))

        if any(term in q for term in ["quote", "recommend", "should we"]):
            if recommendation == "QUOTE":
                answer = (
                    f"Yes. The current underwriting view supports a quote: recommendation is {recommendation}, "
                    f"technical ROL is {technical_rol_pct:.1f}%, and the indicative premium is ${premium:,.0f}."
                )
            else:
                answer = (
                    f"No. The current numbers do not support a quote: recommendation is {recommendation}, "
                    f"technical ROL is {technical_rol_pct:.1f}%."
                )
        elif "premium" in q:
            answer = f"Indicative technical premium is ${premium:,.0f}."
        elif "rol" in q:
            answer = f"Technical ROL is {technical_rol_pct:.1f}% based on the current pricing output."
        else:
            answer = "The pricing agent is using the local underwriting output while the optional notebook-based agent is unavailable."

        return {
            "question": question,
            "answer": answer
        }


# -------------------------------
# RUN DIRECTLY (OPTIONAL)
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.api_main:app", host="127.0.0.1", port=8000, reload=True)