"""Open-Meteo API client: geocoding and forecast time series (no API key required)."""
from typing import Any, Dict, List, Optional, Tuple

import httpx

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def geocode(location: str) -> Optional[Tuple[float, float, str]]:
    """Resolve location name to (latitude, longitude, timezone). Returns None if not found."""
    if not location or len(location.strip()) < 2:
        return None
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(GEOCODE_URL, params={"name": location.strip(), "count": 1})
            r.raise_for_status()
            data = r.json()
    except Exception:
        return None
    results = data.get("results") or []
    if not results:
        return None
    first = results[0]
    lat = float(first.get("latitude", 0))
    lon = float(first.get("longitude", 0))
    tz = first.get("timezone", "auto")
    return (lat, lon, tz)


def fetch_forecast(
    latitude: float,
    longitude: float,
    timezone: str = "auto",
    past_days: int = 0,
    hourly: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Fetch forecast (and optional past days) time series. Returns raw API response."""
    if hourly is None:
        hourly = ["temperature_2m", "precipitation"]
    params: Dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "hourly": ",".join(hourly),
    }
    if past_days:
        params["past_days"] = min(92, past_days)
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.get(FORECAST_URL, params=params)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        return {"error": str(e)}


def extract_timeseries(data: Dict[str, Any]) -> Dict[str, List[Any]]:
    """Extract hourly time and variables from forecast response. Handles missing values."""
    out: Dict[str, List[Any]] = {}
    hourly = data.get("hourly") or {}
    if "error" in data:
        return out
    times = hourly.get("time") or []
    out["time"] = list(times)
    for key in ["temperature_2m", "precipitation"]:
        vals = hourly.get(key)
        if vals is not None:
            out[key] = [v if v is not None else None for v in vals]
    return out
