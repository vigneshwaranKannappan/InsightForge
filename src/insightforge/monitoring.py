"""Lightweight LLMOps monitoring: append every interaction to a JSON-lines log."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from insightforge.assistant import AssistantResponse


class InteractionLogger:
    def __init__(self, log_file: str | Path) -> None:
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(self, response: "AssistantResponse", *, session_id: str, model: str) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "session_id": session_id,
            "model": model,
            "question": response.question,
            "standalone_question": response.standalone_question,
            "source_topics": [d.metadata.get("topic") for d in response.sources],
            "answer_chars": len(response.answer),
            "latency_seconds": response.latency_seconds,
        }
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def load(self) -> pd.DataFrame:
        if not self.log_file.exists():
            return pd.DataFrame()
        with self.log_file.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        return pd.DataFrame(rows)


def summarize_interactions(log: pd.DataFrame) -> dict[str, float]:
    if log.empty:
        return {"interactions": 0, "sessions": 0, "avg_latency": 0.0, "p95_latency": 0.0}
    return {
        "interactions": len(log),
        "sessions": int(log["session_id"].nunique()),
        "avg_latency": float(log["latency_seconds"].mean()),
        "p95_latency": float(log["latency_seconds"].quantile(0.95)),
    }
