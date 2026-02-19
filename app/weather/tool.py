"""Weather time-series tool: geocode -> fetch Open-Meteo -> run sandboxed analytics -> summary."""
from typing import Any, Dict, Optional

from app.weather.open_meteo import (
    extract_timeseries,
    fetch_forecast,
    geocode,
)
from app.weather.analytics_runner import run_analytics_sandbox


def get_weather_analysis(
    location: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """
    Fetch weather time series for a location, run analytics in sandbox, return human-readable summary.
    Dates optional; default is next 7 days forecast.
    """
    if not location or not location.strip():
        return "Error: Please provide a location (e.g. city name)."

    coords = geocode(location.strip())
    if not coords:
        return f"Could not find coordinates for: {location}"

    lat, lon, tz = coords
    data = fetch_forecast(latitude=lat, longitude=lon, timezone=tz, past_days=0)
    if data.get("error"):
        return f"Failed to fetch weather data: {data['error']}"

    ts = extract_timeseries(data)
    if not ts or not ts.get("time"):
        return "No time series data returned from the API."

    analytics = run_analytics_sandbox(ts)
    if analytics.get("error"):
        return f"Weather analytics error: {analytics['error']}"

    # Build summary
    parts = [f"Weather time series for {location} (lat={lat:.2f}, lon={lon:.2f}):"]
    if analytics.get("total_points"):
        parts.append(f"- Total hourly points: {analytics['total_points']}")
    if analytics.get("temp_min") is not None and analytics.get("temp_max") is not None:
        parts.append(f"- Temperature range: {analytics['temp_min']} deg C to {analytics['temp_max']} deg C")
    if analytics.get("rolling_mean_last_24h") is not None:
        parts.append(f"- 24h rolling mean (latest): {analytics['rolling_mean_last_24h']} deg C")
    if analytics.get("volatility_c") is not None:
        parts.append(f"- Volatility (std dev): {analytics['volatility_c']} deg C")
    if analytics.get("missingness_count") is not None:
        parts.append(f"- Missing values: {analytics['missingness_count']}")
    if analytics.get("anomaly_flags") is not None and analytics["anomaly_flags"] > 0:
        parts.append(f"- Anomaly flags (points >2 std from mean): {analytics['anomaly_flags']}")

    return "\n".join(parts)
