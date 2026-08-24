import pickle
from pathlib import Path

import numpy as np


MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"

_model = None


class ProfitabilityModel:
    def __init__(self, model_data):
        self.model = model_data["model"]
        self.features = model_data["features"]

    def predict(
        self,
        revenue: float,
        opex: float,
        marketing: float,
        horizon_months: int,
    ) -> dict:
        features = np.array([
            [revenue, opex, marketing]
        ])

        prediction = self.model.predict(features)[0]

        return {
            "predicted_net_profit_margin": float(prediction)
        }


def get_model() -> ProfitabilityModel:
    global _model

    if _model is None:
        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)

        _model = ProfitabilityModel(model_data)

    return _model