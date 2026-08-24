from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    expected_monthly_revenue: float = Field(gt=0)
    fixed_operational_expenses: float = Field(ge=0)
    marketing_budget: float = Field(ge=0)
    horizon_months: int = Field(ge=1)


class ForecastResponse(BaseModel):
    predicted_net_profit_margin: float