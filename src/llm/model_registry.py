"""Model registry — available Ollama models + capability tags."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    name: str
    display_name: str
    size_gb: float
    vram_gb: int
    best_for: str
    is_default: bool = False
    supports_web_search: bool = False
    partial_offload: bool = False  # can run with split GPU/CPU layers


# Ordered by preference (first = recommended)
MODELS: list[ModelInfo] = [
    ModelInfo(
        name="crow-9b-opus",
        display_name="Crow-9B-Opus-4.6-Distill-Heretic (Qwen3.5 9B) — recommended",
        size_gb=6.0,
        vram_gb=4,
        best_for="Field-context analysis, web-search-grounded reasoning",
        is_default=True,
        supports_web_search=True,
        partial_offload=True,
    ),
    ModelInfo(
        name="deepseek-r1:14b",
        display_name="DeepSeek-R1 14B",
        size_gb=9.0,
        vram_gb=10,
        best_for="Reasoning / gap analysis (partial offload on 6 GB VRAM)",
        partial_offload=True,
    ),
    ModelInfo(
        name="qwen2.5:14b",
        display_name="Qwen 2.5 14B",
        size_gb=9.0,
        vram_gb=10,
        best_for="Balanced reasoning + speed (partial offload on 6 GB VRAM)",
        partial_offload=True,
    ),
    ModelInfo(
        name="phi4:14b",
        display_name="Phi-4 14B",
        size_gb=9.0,
        vram_gb=10,
        best_for="Structured JSON extraction",
        partial_offload=True,
    ),
    ModelInfo(
        name="mistral:7b",
        display_name="Mistral 7B",
        size_gb=4.1,
        vram_gb=6,
        best_for="Fast summaries, low VRAM / CPU fallback",
    ),
]

# Fallback order
FALLBACK_ORDER = [
    "crow-9b-opus",
    "deepseek-r1:14b",
    "qwen2.5:14b",
    "phi4:14b",
    "mistral:7b",
]


def get_default() -> str:
    """Return the default model name."""
    return "crow-9b-opus"


def get_model_info(name: str) -> ModelInfo | None:
    for m in MODELS:
        if m.name == name:
            return m
    return None


def get_all_models() -> list[ModelInfo]:
    return MODELS.copy()


def get_field_context_model() -> str:
    """Return the model name designated for field-context deep analysis."""
    return "crow-9b-opus"


def get_field_context_model_info() -> ModelInfo:
    """Return ModelInfo for the field-context model."""
    return get_model_info("crow-9b-opus") or MODELS[-1]


async def find_best_available(installed: list[str]) -> str | None:
    """Given list of installed model names, return best match from fallback order."""
    installed_lower = {m.lower() for m in installed}
    for candidate in FALLBACK_ORDER:
        if any(candidate in m for m in installed_lower):
            return candidate
    return None
