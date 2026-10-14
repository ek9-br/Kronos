"""Contract checks for direct OHLCV forecasts through the existing Flask API."""

import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest


APP_PATH = Path(__file__).resolve().parents[1] / "webui/app.py"


def load_app():
    spec = importlib.util.spec_from_file_location("kronos_native_api_test", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candles(timeframe="1h", count=4):
    hours = {"1h": 1, "2h": 2, "4h": 4, "1D": 24}[timeframe]
    start = datetime(2026, 10, 14, 10, tzinfo=timezone(timedelta(hours=-3)))
    return [
        {
            "timestamp": (start + timedelta(hours=hours * i)).isoformat(),
            "open": 100 + i,
            "high": 102 + i,
            "low": 99 + i,
            "close": 101 + i,
            "volume": 10 + i,
        }
        for i in range(count)
    ]


class Predictor:
    def __init__(self):
        self.calls = 0
        self.input_times = None
        self.output_times = None

    def predict(self, df, x_timestamp, y_timestamp, pred_len, **kwargs):
        self.calls += 1
        self.input_times = list(x_timestamp)
        self.output_times = list(y_timestamp)
        return pd.DataFrame({
            "open": [105.0] * pred_len,
            "high": [107.0] * pred_len,
            "low": [104.0] * pred_len,
            "close": [106.0] * pred_len,
            "volume": [12.0] * pred_len,
        })


def client_with_model(monkeypatch):
    module = load_app()
    predictor = Predictor()
    monkeypatch.setattr(module, "MODEL_AVAILABLE", True)
    monkeypatch.setattr(module, "predictor", predictor)
    monkeypatch.setattr(module, "current_model_key", "kronos-small", raising=False)
    return module, predictor, module.app.test_client()


@pytest.mark.parametrize("timeframe,hours", [("1h", 1), ("2h", 2), ("4h", 4), ("1D", 24)])
def test_future_json_contract_and_utc_timestamps(timeframe, hours, monkeypatch):
    module, predictor, client = client_with_model(monkeypatch)
    monkeypatch.setattr(module, "create_prediction_chart", lambda *args: pytest.fail("chart generated"))
    monkeypatch.setattr(module, "save_prediction_results", lambda **kwargs: pytest.fail("file saved"))
    response = client.post("/api/predict", json={
        "symbol": "BTC-USDT-USDT", "timeframe": timeframe, "mode": "future",
        "lookback": 4, "pred_len": 2, "candles": candles(timeframe),
    })
    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["success"] is True
    assert (body["symbol"], body["timeframe"], body["model"]) == ("BTC-USDT-USDT", timeframe, "Kronos-small")
    assert (body["lookback"], body["pred_len"]) == (4, 2)
    assert body["last_historical_timestamp"] == predictor.input_times[-1].isoformat()
    assert body["last_historical_timestamp"].endswith("+00:00")
    expected = [(predictor.input_times[-1] + pd.Timedelta(hours=hours * i)).isoformat() for i in (1, 2)]
    assert [row["timestamp"] for row in body["prediction"]["candles"]] == expected
    assert [t.isoformat() for t in predictor.output_times] == expected
    assert body["prediction"]["predicted_return_percent"] == pytest.approx(100 * (106 / 104 - 1))
    assert body["prediction"]["direction"] == "UP"
    assert body["inference_ms"] >= 0
    assert "chart" not in body
    assert predictor.calls == 1


@pytest.mark.parametrize("change,error", [
    (lambda c: c[1].update(timestamp=c[0]["timestamp"]), "increasing"),
    (lambda c: c[1].update(timestamp=c[2]["timestamp"]), "increasing"),
    (lambda c: c[2].update(timestamp="2026-10-14T15:30:00Z"), "timeframe"),
    (lambda c: c[1].update(high=99), "OHLC"),
    (lambda c: c[1].update(close=0), "positive"),
    (lambda c: c[1].update(volume=-1), "non-negative"),
    (lambda c: c.pop(), "Insufficient"),
])
def test_invalid_json_is_rejected_before_inference(change, error, monkeypatch):
    _, predictor, client = client_with_model(monkeypatch)
    history = candles()
    change(history)
    response = client.post("/api/predict", json={"timeframe": "1h", "lookback": 4, "pred_len": 2, "candles": history})
    assert response.status_code == 400, response.get_json()
    assert error.lower() in response.get_json()["error"].lower()
    assert predictor.calls == 0


def test_json_requires_supported_timeframe(monkeypatch):
    _, predictor, client = client_with_model(monkeypatch)
    response = client.post("/api/predict", json={"lookback": 4, "pred_len": 2, "candles": candles()})
    assert response.status_code == 400
    assert "timeframe" in response.get_json()["error"]
    assert predictor.calls == 0


def test_output_ohlcv_is_consistent_without_changing_open_or_close(monkeypatch):
    module, predictor, client = client_with_model(monkeypatch)
    def inconsistent(df, x_timestamp, y_timestamp, pred_len, **kwargs):
        predictor.calls += 1
        return pd.DataFrame({"open": [105.0] * pred_len, "high": [104.0] * pred_len,
                             "low": [108.0] * pred_len, "close": [106.0] * pred_len,
                             "volume": [-2.0] * pred_len})
    monkeypatch.setattr(predictor, "predict", inconsistent)
    response = client.post("/api/predict", json={"timeframe": "1h", "lookback": 4, "pred_len": 2, "candles": candles()})
    assert response.status_code == 200, response.get_json()
    row = response.get_json()["prediction"]["candles"][0]
    assert row == {"timestamp": "2026-10-14T17:00:00+00:00", "open": 105.0,
                   "high": 106.0, "low": 105.0, "close": 106.0, "volume": 0.0}


@pytest.mark.parametrize("parameter,value", [("lookback", "abc"), ("pred_len", 0),
                                              ("temperature", 0), ("top_p", 2),
                                              ("sample_count", 0)])
def test_invalid_parameters_are_client_errors(parameter, value, monkeypatch):
    _, predictor, client = client_with_model(monkeypatch)
    payload = {"timeframe": "1h", "lookback": 4, "pred_len": 2, "candles": candles()}
    payload[parameter] = value
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400, response.get_json()
    assert predictor.calls == 0
