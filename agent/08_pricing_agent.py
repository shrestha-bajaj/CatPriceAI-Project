from pathlib import Path
import pandas as pd
import sqlite3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "catprice.db"
PRICING_PATH = DATA_DIR / "pricing_output.csv"
BURNING_COST_PATH = DATA_DIR / "burning_cost.csv"
EXPOSURE_PATH = DATA_DIR / "exposure_summary.csv"
LOSS_SUMMARY_PATH = DATA_DIR / "loss_summary.csv"


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _read_db_table(table_name: str) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)


class PricingAgent:
    def __init__(self):
        if not DB_PATH.exists():
            raise FileNotFoundError(f"Database not found: {DB_PATH}")

        self.pricing = _read_csv(PRICING_PATH)
        self.burning_cost = _read_csv(BURNING_COST_PATH)
        self.exposure = _read_csv(EXPOSURE_PATH)
        self.loss_summary = _read_csv(LOSS_SUMMARY_PATH)
        self.pricing_row = self.pricing.iloc[0].to_dict()

    def _build_openai_agent_prompt(self, question: str) -> str:
        lines = [
            "You are an underwriting pricing assistant. Use the available pricing data and exposure summary to answer the question.",
            "Do not invent values; use only the metrics provided below.",
            "",
            "Pricing data:",
        ]
        for key, value in self.pricing_row.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
        exposure_row = self.exposure.iloc[0].to_dict()
        lines.append("Exposure summary:")
        for key, value in exposure_row.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
        lines.append(f"Question: {question}")
        lines.append("Answer concisely.")
        return "\n".join(lines)

    def _openai_answer(self, question: str, api_key: str) -> str:
        try:
            import openai

            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a precise underwriting assistant."},
                    {"role": "user", "content": self._build_openai_agent_prompt(question)},
                ],
                max_tokens=250,
                temperature=0.0,
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            return f"OpenAI pricing agent error: {exc}. Using local router instead."

    def tool_get_technical_price(self) -> str:
        technical_rol = float(self.pricing_row.get("technical_rol_pct", self.pricing_row.get("technical_rol", 0.0) * 100))
        premium = float(self.pricing_row.get("technical_premium_usd", self.pricing_row.get("technical_premium", 0.0)))
        recommendation = str(self.pricing_row.get("recommendation", "")).upper()
        return (
            f"Technical ROL: {technical_rol:.2f}%. "
            f"Indicative technical premium: ${premium:,.0f}. "
            f"Recommendation: {recommendation}."
        )

    def tool_get_loss_history(self) -> str:
        top_losses = self.loss_summary.sort_values("annual_gross_loss", ascending=False).head(5)
        rows = []
        for _, row in top_losses.iterrows():
            rows.append(f"{int(row['year'])}: ${row['annual_gross_loss']:,.0f} gross")
        return "Top 5 annual loss years:\n" + "\n".join(rows)

    def tool_get_market_comparison(self) -> str:
        market_rol = float(self.pricing_row.get("market_benchmark_rol", self.pricing_row.get("market_rol_pct", 0.0)))
        technical_rol = float(self.pricing_row.get("technical_rol_pct", self.pricing_row.get("technical_rol", 0.0) * 100))
        adequacy = float(self.pricing_row.get("adequacy_vs_market_pct", 0.0))
        return (
            f"Technical ROL is {technical_rol:.2f}%, market benchmark ROL is {market_rol:.2f}%. "
            f"Adequacy versus market is {adequacy:.2f}% above market."
        )

    def tool_get_portfolio_concentration(self) -> str:
        row = self.exposure.iloc[0]
        state = row.get("highest_tiv_state", "Unknown")
        state_pct = float(row.get("highest_state_tiv_share_pct", 0.0))
        coastal_pct = float(row.get("coastal_tiv_pct", 0.0))
        flood_pct = float(row.get("high_risk_flood_tiv_pct", 0.0))
        return (
            f"Top state is {state} with {state_pct:.2f}% of TIV. "
            f"{coastal_pct:.2f}% of TIV is within 50km of the coast and {flood_pct:.2f}% is in high-risk flood zones."
        )

    def answer(self, question: str) -> str:
        q = question.lower()
        if any(term in q for term in ["technical price", "technical rol", "price", "premium"]):
            return self.tool_get_technical_price()
        if any(term in q for term in ["loss history", "top losses", "largest loss", "loss years"]):
            return self.tool_get_loss_history()
        if any(term in q for term in ["market", "benchmark", "adequacy"]):
            return self.tool_get_market_comparison()
        if any(term in q for term in ["concentration", "state", "coastal", "flood zone"]):
            return self.tool_get_portfolio_concentration()
        if any(term in q for term in ["should we quote", "quote", "pass"]):
            recommendation = str(self.pricing_row.get("recommendation", "")).upper()
            technical_rol = float(self.pricing_row.get("technical_rol_pct", 0.0))
            answer = (
                f"The current recommendation is {recommendation}. "
                f"Technical ROL is {technical_rol:.2f}%.")
            if recommendation == "QUOTE":
                return answer + " This supports moving forward with a quote."
            return answer + " This suggests a pass."
        return (
            "I can answer pricing questions about technical ROL, premium, market adequacy, "
            "portfolio exposure, and loss history. Please ask one of those."
        )
