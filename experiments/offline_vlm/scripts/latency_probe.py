"""Time one offline alignment call on already uploaded files, at a given media resolution.

Usage: python experiments/offline_vlm/scripts/latency_probe.py <report.json> <low|medium|high>
Never prints the key.
"""

import json
import sys
from pathlib import Path
from time import perf_counter

from dotenv import dotenv_values
from google import genai

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from src.guideme.offline_vlm import predict_offline_alignment  # noqa: E402

report = json.loads(Path(sys.argv[1]).read_text())
level = sys.argv[2]
key = dotenv_values(REPO / ".env").get("GEMINI_API_KEY")
client = genai.Client(api_key=key, http_options={"timeout": 900000, "retry_options": {"attempts": 1}})
started = perf_counter()
try:
    result = predict_offline_alignment(
        client, report["uploads"]["reference"]["uri"], report["uploads"]["execution"]["uri"],
        report["checklist"]["rows"], execution_duration_s=report["inputs"]["execution_clip"]["duration_s"],
        fps=1, model="gemini-3.5-flash-lite", media_resolution=f"MEDIA_RESOLUTION_{level.upper()}",
    )
    usage = result["response"].get("usage_metadata") or {}
    out = {"level": level, "seconds": round(perf_counter() - started, 1),
           "valid": result["prediction"] is not None, "validation_error": result["validation_error"],
           "prompt_tokens": usage.get("prompt_token_count"),
           "output_tokens": usage.get("candidates_token_count"),
           "thinking_tokens": usage.get("thoughts_token_count"),
           "prediction": result["prediction"], "raw_text": (result["response"].get("candidates") or [{}])[0].get("content", {}).get("parts", [{}])[0].get("text")}
except Exception as error:
    out = {"level": level, "seconds": round(perf_counter() - started, 1), "error": type(error).__name__}
print(json.dumps(out, indent=1))
