"""Sentence splitting and BERT-based encoding utilities."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

_SENTENCE_REGEX = re.compile(r"[^。！？!?]*[。！？!?]|[^。！？!?]+$")


@dataclass
class EncoderConfig:
    model_name: str = "hfl/chinese-bert-wwm-ext"
    pooling: str = "mean"  # "mean" or "cls"
    max_length: int = 512
    batch_size: int = 8
    device: str | None = None


class BertSentenceEncoder:
    """Encodes sentences using an off-the-shelf Chinese BERT."""

    def __init__(self, config: EncoderConfig) -> None:
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name)
        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Splits text into sentences using Chinese punctuation heuristics."""
        text = text.strip()
        if not text:
            return []
        text = re.sub(r"\s+", " ", text)
        parts = _SENTENCE_REGEX.findall(text)
        return [part.strip() for part in parts if part and part.strip()]

    def encode(self, sentences: Sequence[str]) -> np.ndarray:
        if not sentences:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        outputs: List[np.ndarray] = []
        batch_size = max(1, self.config.batch_size)
        pooling = self.config.pooling.lower()
        with torch.no_grad():
            for start in range(0, len(sentences), batch_size):
                batch = sentences[start : start + batch_size]
                encoded = self.tokenizer(
                    list(batch),
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                model_out = self.model(**encoded)
                hidden_states = model_out.last_hidden_state  # (batch, seq, hidden)
                attention_mask = encoded["attention_mask"].unsqueeze(-1)
                if pooling == "cls":
                    pooled = hidden_states[:, 0]
                else:
                    masked = hidden_states * attention_mask
                    summed = masked.sum(dim=1)
                    counts = attention_mask.sum(dim=1).clamp(min=1)
                    pooled = summed / counts
                outputs.append(pooled.cpu().numpy())
        return np.vstack(outputs)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        return embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms
