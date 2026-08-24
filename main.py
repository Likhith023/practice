import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from schemas import ForecastRequest, ForecastResponse
from ml_model import get_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fintrack-api")

app = FastAPI(
    title="FinTrack Profitability API",
    description="Serves ML-based net-profit-margin forecasts for the FinTrack dashboard.",
    version="1.0.0",
)

# Dashboard is served from a different origin during local dev (Live Server,
# Vite, etc). Lock this down to your real frontend origin(s) in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "FinTrack Profitability API is running",
        "docs": "/docs",
        "health": "/health",
    }
@app.on_event("startup")
def load_model() -> None:
    # Trains/loads the model once when the server boots, not per-request.
    get_model()
    logger.info("Profitability model ready.")


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/api/v1/predict-profitability", response_model=ForecastResponse)
def predict_profitability(payload: ForecastRequest) -> ForecastResponse:
    """
    Matches the frontend's fetch('/api/v1/predict-profitability', { ... })
    call in mlForecastForm's submit handler. Input/output field names line up
    exactly with what dashboard.js sends and reads.
    """
    try:
        model = get_model()
        result = model.predict(
            revenue=payload.expected_monthly_revenue,
            opex=payload.fixed_operational_expenses,
            marketing=payload.marketing_budget,
            horizon_months=payload.horizon_months,
        )
        return ForecastResponse(**result)

    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    