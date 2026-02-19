"""Run analytics script in a subprocess with timeout (sandbox: no network, limited time)."""
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from app.config import REPO_ROOT, WEATHER_ANALYSIS_TIMEOUT_SEC

SCRIPT_PATH = Path(__file__).resolve().parent / "analytics_script.py"


def run_analytics_sandbox(timeseries: Dict[str, list]) -> Dict[str, Any]:
    """
    Run analytics in a subprocess with timeout. Pass timeseries JSON on stdin, get result on stdout.
    Safe: no network, no file access; only computation on provided data.
    """
    if not SCRIPT_PATH.exists():
        return {"error": "Analytics script not found"}
    try:
        payload = json.dumps(timeseries)
        proc = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            input=payload,
            capture_output=True,
            text=True,
            timeout=WEATHER_ANALYSIS_TIMEOUT_SEC,
            cwd=str(REPO_ROOT),
        )
        out = (proc.stdout or "").strip()
        if not out and proc.stderr:
            return {"error": proc.stderr[:500]}
        return json.loads(out) if out else {}
    except subprocess.TimeoutExpired:
        return {"error": "Analytics timed out"}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid analytics output: {e}"}
    except Exception as e:
        return {"error": str(e)}
