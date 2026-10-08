"""Real Open-Meteo weather only; never substitute simulated readings."""
import math
import requests
import pandas as pd

CITIES = {
    "Chennai": (13.0827, 80.2707), "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946), "Hyderabad": (17.3850, 78.4867),
    "Kolkata": (22.5726, 88.3639),
}


def weather_description(code):
    if code == 0:
        return "Clear sky"
    if code in (1, 2, 3):
        return "Mainly clear / partly cloudy / overcast"
    if code in (45, 48):
        return "Fog"
    if code in (51, 53, 55, 56, 57):
        return "Drizzle"
    if code in (61, 63, 65, 66, 67, 80, 81, 82):
        return "Rain / showers"
    if code in (71, 73, 75, 77, 85, 86):
        return "Snow"
    if code in (95, 96, 99):
        return "Thunderstorm"
    return f"Weather code {code}"


def fetch_weather(city):
    latitude, longitude = CITIES[city]
    try:
        response = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": latitude, "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,rain,weather_code",
            "hourly": "rain", "forecast_days": 2, "timezone": "Asia/Kolkata",
        }, timeout=20)
        response.raise_for_status()
        payload = response.json()
        current = payload["current"]
        required = ["temperature_2m", "relative_humidity_2m", "rain", "weather_code", "time"]
        if any(current.get(key) is None for key in required):
            raise ValueError("Weather response is missing current measurements")
        for key in required[:-1]:
            if not isinstance(current[key], (int, float)) or not math.isfinite(current[key]):
                raise ValueError("Weather response contains invalid numeric measurements")
        interval = current.get("interval", 900)
        if not isinstance(interval, (int, float)) or not math.isfinite(interval) or interval <= 0:
            raise ValueError("Weather response contains an invalid rainfall interval")
        if current["rain"] < 0 or not 0 <= current["relative_humidity_2m"] <= 100:
            raise ValueError("Weather response contains measurements outside valid ranges")
        hourly = payload.get("hourly", {})
        forecast = pd.DataFrame({"time": hourly.get("time", []), "rainfall": hourly.get("rain", [])})
        return current, forecast
    except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError) as error:
        raise RuntimeError(f"Open-Meteo weather is unavailable: {error}") from error
