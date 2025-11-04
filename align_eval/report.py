"""Report helpers for AlignEval."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

from .metrics import SentenceMetrics


@dataclass
class BookReport:
    book_id: str
    global_metrics: Dict[str, float]
    sentences: Sequence[SentenceMetrics]
    alignment_path: Sequence[Sequence[int]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "book_id": self.book_id,
            "global_metrics": self.global_metrics,
            "alignment_path": [list(pair) for pair in self.alignment_path],
            "sentences": [asdict(sent) for sent in self.sentences],
        }


def write_report(path: str | Path, summary: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
