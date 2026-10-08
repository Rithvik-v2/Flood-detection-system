"""Run with python -m unittest -v. Tests use temporary history, never user data."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import joblib
import pandas as pd
import requests
from streamlit.testing.v1 import AppTest
import database
from train_model import FEATURES, SCENARIOS, ROOT
from weather import fetch_weather


class ProjectTests(unittest.TestCase):
    def test_saved_model_scenarios(self):
        model = joblib.load(ROOT / "flood_model.pkl")
        for (name, values), expected in zip(SCENARIOS.items(), ["LOW", "MEDIUM", "HIGH"]):
            self.assertEqual(model.predict(pd.DataFrame([values], columns=FEATURES))[0], expected, name)

    def test_database_roundtrip_and_clear(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "history.db"
            self.assertTrue(database.get_history(path).empty)
            for values, risk in zip(SCENARIOS.values(), ["LOW", "MEDIUM", "HIGH"]):
                database.save_prediction(values, risk, path=path)
            history = database.get_history(path)
            self.assertEqual(len(history), 3)
            self.assertIn("WARNING", history.iloc[0].alert)
            self.assertEqual(history.iloc[0].water_level, 7.5)
            database.clear_history(path)
            self.assertTrue(database.get_history(path).empty)

    def test_weather_failure(self):
        with patch("weather.requests.get", side_effect=requests.ConnectionError("offline")):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                fetch_weather("Chennai")

    def test_weather_response(self):
        from unittest.mock import Mock
        response = Mock()
        response.json.return_value = {
            "current": {"temperature_2m": 28, "relative_humidity_2m": 80, "rain": 2,
                        "weather_code": 61, "time": "2026-10-08T12:00", "interval": 900},
            "hourly": {"time": ["2026-10-08T12:00"], "rain": [3]},
        }
        with patch("weather.requests.get", return_value=response):
            current, forecast = fetch_weather("Chennai")
            self.assertEqual(current["rain"], 2)
            self.assertEqual(forecast.iloc[0].rainfall, 3)
            response.raise_for_status.assert_called_once()

    def test_dashboard_flow(self):
        # AppTest runs the real Streamlit script and widgets using isolated SQLite.
        original_get, original_save, original_clear = database.get_history, database.save_prediction, database.clear_history
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "app.db"
            with patch("database.get_history", side_effect=lambda: original_get(path)), \
                 patch("database.save_prediction", side_effect=lambda values, risk, source: original_save(values, risk, source, path)), \
                 patch("database.clear_history", side_effect=lambda: original_clear(path)), \
                 patch("weather.requests.get", side_effect=requests.ConnectionError("offline")):
                app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
                self.assertFalse(app.exception)
                app.sidebar.radio[0].set_value("Flood Prediction").run()
                for name, expected in zip(SCENARIOS, ["LOW", "MEDIUM", "HIGH"]):
                    next(button for button in app.button if button.label == name).click().run()
                    next(button for button in app.button if button.label == "Predict Flood Risk").click().run()
                    self.assertFalse(app.exception)
                    self.assertEqual(original_get(path).iloc[0].risk, expected)
                app.sidebar.radio[0].set_value("Home Dashboard").run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.metric), 4)
                app.sidebar.radio[0].set_value("Prediction History").run()
                self.assertFalse(app.exception)
                app.checkbox[0].check().run()
                next(button for button in app.button if button.label == "Clear demo history").click().run()
                self.assertTrue(original_get(path).empty)
                app.sidebar.radio[0].set_value("About Project").run()
                self.assertFalse(app.exception)
                app.sidebar.radio[0].set_value("Live Weather").run()
                self.assertFalse(app.exception)
                self.assertTrue(app.error)

    def test_weather_transfer(self):
        from unittest.mock import Mock
        response = Mock()
        response.json.return_value = {
            "current": {"temperature_2m": 27., "relative_humidity_2m": 82., "rain": 3.,
                        "weather_code": 61, "time": "2026-10-08T12:00", "interval": 900},
            "hourly": {"time": ["2026-10-08T12:00"], "rain": [3.]},
        }
        original_get, original_save = database.get_history, database.save_prediction
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weather.db"
            with patch("database.get_history", side_effect=lambda: original_get(path)), \
                 patch("database.save_prediction", side_effect=lambda values, risk, source: original_save(values, risk, source, path)), \
                 patch("weather.requests.get", return_value=response):
                app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
                app.sidebar.radio[0].set_value("Live Weather").run()
                self.assertFalse(app.exception)
                next(b for b in app.button if b.label == "Use Current Weather for Prediction").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(app.sidebar.radio[0].value, "Flood Prediction")
                self.assertEqual(app.number_input[0].value, 12.)
                next(b for b in app.button if b.label == "Predict Flood Risk").click().run()
                self.assertTrue(original_get(path).empty)
                self.assertTrue(app.error)
                next(n for n in app.number_input if "Water level" in n.label).set_value(2.)
                app.checkbox[0].check()
                next(b for b in app.button if b.label == "Predict Flood Risk").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(original_get(path).iloc[0].water_level, 2.)
                self.assertIn("Open-Meteo", original_get(path).iloc[0].source)


if __name__ == "__main__":
    unittest.main()
