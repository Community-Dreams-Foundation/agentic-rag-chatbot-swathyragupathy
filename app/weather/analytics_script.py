#!/usr/bin/env python3
"""
Sandboxed analytics: reads JSON from stdin, computes rolling mean, volatility,
missingness, and simple anomaly flags; prints JSON to stdout.
No network, no file I/O beyond stdin/stdout. Run with subprocess and timeout.
"""
import json
import math
import sys


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception as e:
        print(json.dumps({"error": f"Invalid JSON: {e}"}))
        sys.exit(1)

    time_arr = payload.get("time") or []
    temp_arr = payload.get("temperature_2m") or []
    prec_arr = payload.get("precipitation") or []

    # Align length
    n = min(len(time_arr), len(temp_arr)) if time_arr and temp_arr else 0
    if n == 0:
        print(json.dumps({"error": "No time series data", "rolling_mean": None, "volatility": None, "missingness": 0, "anomaly_flags": 0}))
        return

    temps = []
    for i in range(n):
        v = temp_arr[i] if i < len(temp_arr) else None
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            temps.append(float(v))
        else:
            temps.append(None)

    # Rolling mean (window 24 hours)
    window = min(24, len(temps))
    rolling_means = []
    for i in range(len(temps)):
        start = max(0, i - window + 1)
        chunk = [t for t in temps[start : i + 1] if t is not None]
        if chunk:
            rolling_means.append(round(sum(chunk) / len(chunk), 2))
        else:
            rolling_means.append(None)

    # Volatility (std of temperature)
    valid = [t for t in temps if t is not None]
    if len(valid) >= 2:
        mean_t = sum(valid) / len(valid)
        variance = sum((x - mean_t) ** 2 for x in valid) / (len(valid) - 1)
        volatility = round(math.sqrt(variance), 2)
    else:
        volatility = None

    # Missingness
    missingness = sum(1 for t in temps if t is None)

    # Simple anomaly: count points > 2 std from mean
    anomaly_flags = 0
    if len(valid) >= 3 and volatility is not None and volatility > 0:
        mean_t = sum(valid) / len(valid)
        for t in valid:
            if abs(t - mean_t) > 2 * volatility:
                anomaly_flags += 1

    result = {
        "rolling_mean_last_24h": rolling_means[-1] if rolling_means else None,
        "rolling_mean_sample": rolling_means[-5:] if len(rolling_means) >= 5 else rolling_means,
        "volatility_c": volatility,
        "missingness_count": missingness,
        "total_points": n,
        "anomaly_flags": anomaly_flags,
        "temp_min": round(min(valid), 2) if valid else None,
        "temp_max": round(max(valid), 2) if valid else None,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
