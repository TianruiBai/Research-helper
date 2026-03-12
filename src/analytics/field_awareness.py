"""Field-aware pace & context detection.

Different research domains evolve at vastly different speeds:
  - CS / AI: very fast (6-12 month cycles, rapid obsolescence)
  - Biomedical / clinical: slow (multi-year trials, regulatory gates)
  - Physics / math: moderate (theory-driven, less trend churn)
  - Social science / humanities: slow to moderate

This module detects the likely domain from query keywords and adjusts
scoring weights and interpretation accordingly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Domain keyword → field category mapping
# Order matters: first match wins
_FIELD_PATTERNS: list[tuple[str, str]] = [
    # Fast-moving tech fields
    (r"\b(LLM|GPT|transformer|diffusion model|deep learning|neural network|"
     r"machine learning|reinforcement learning|NLP|computer vision|"
     r"generative AI|large language model|GAN|autoencoder|BERT|"
     r"speech recognition|autonomous driving|robotics|cyber|blockchain|"
     r"cloud computing|edge computing|IoT|5G|6G|quantum computing|"
     r"federated learning|foundation model)\b", "cs_fast"),
    # Moderate-pace engineering / applied CS
    (r"\b(software engineering|database|information retrieval|HCI|"
     r"sensor|embedded system|network|signal processing|FPGA|VLSI|"
     r"control system|optimization|algorithm|compiler|operating system)\b", "engineering"),
    # Biomedical / clinical (slow, evidence-based)
    (r"\b(clinical trial|drug discovery|pharmaceutical|genomics|"
     r"proteomics|cancer|tumor|oncology|immunotherapy|CRISPR|"
     r"vaccine|pathology|epidemiology|biomarker|FDA|EMA|"
     r"randomized controlled|placebo|cohort study|meta-analysis|"
     r"cardiology|neurology|radiology|surgery|patient|diagnosis|"
     r"therapeutic|treatment|disease|syndrome|receptor|protein|gene)\b", "biomedical"),
    # Physics / math / theoretical
    (r"\b(quantum mechanics|particle physics|cosmology|astrophysics|"
     r"condensed matter|string theory|topology|algebra|number theory|"
     r"differential equation|statistical mechanics|thermodynamics|"
     r"material science|superconductor|semiconductor|lattice|"
     r"plasma physics|optics|photonics)\b", "physics_math"),
    # Environmental / earth science
    (r"\b(climate change|sustainability|renewable energy|solar cell|"
     r"battery|carbon capture|biodiversity|ecology|hydrology|"
     r"atmospheric|ocean|geoscience|pollution|emission)\b", "environmental"),
    # Social science / humanities
    (r"\b(psychology|sociology|economics|political science|education|"
     r"linguistics|ethics|philosophy|anthropology|history|law|policy)\b", "social_science"),
]


@dataclass(frozen=True)
class FieldProfile:
    """Characterises a research field's pace and scoring context."""
    field_category: str          # e.g. "cs_fast", "biomedical"
    display_name: str            # human-readable label
    pace: str                    # "fast", "moderate", "slow"
    typical_cycle_years: float   # how many years for a "generation" of work
    # Weight adjustments for comprehensive score (multipliers on base weights)
    weight_interest: float       # fast fields: interest matters more
    weight_motivation: float     # slow fields: deeper motivation analysis
    weight_confidence: float     # biomedical: confidence is paramount
    weight_market: float         # tech fields: market signal is relevant
    weight_sentiment: float      # all fields: public discourse weight
    # Thresholds for scoring interpretation
    growth_rate_ceiling: float   # what growth rate → max score (% / year)
    citation_velocity_ceiling: float  # what velocity → high impact


_PROFILES: dict[str, FieldProfile] = {
    "cs_fast": FieldProfile(
        field_category="cs_fast",
        display_name="Computer Science / AI (fast-moving)",
        pace="fast",
        typical_cycle_years=1.0,
        weight_interest=0.30,
        weight_motivation=0.10,
        weight_confidence=0.15,
        weight_market=0.30,
        weight_sentiment=0.15,
        growth_rate_ceiling=50.0,
        citation_velocity_ceiling=20.0,
    ),
    "engineering": FieldProfile(
        field_category="engineering",
        display_name="Engineering / Applied CS",
        pace="moderate",
        typical_cycle_years=2.0,
        weight_interest=0.25,
        weight_motivation=0.15,
        weight_confidence=0.20,
        weight_market=0.25,
        weight_sentiment=0.15,
        growth_rate_ceiling=30.0,
        citation_velocity_ceiling=15.0,
    ),
    "biomedical": FieldProfile(
        field_category="biomedical",
        display_name="Biomedical / Clinical Science",
        pace="slow",
        typical_cycle_years=5.0,
        weight_interest=0.15,
        weight_motivation=0.20,
        weight_confidence=0.30,
        weight_market=0.20,
        weight_sentiment=0.15,
        growth_rate_ceiling=15.0,
        citation_velocity_ceiling=8.0,
    ),
    "physics_math": FieldProfile(
        field_category="physics_math",
        display_name="Physics / Mathematics",
        pace="moderate",
        typical_cycle_years=3.0,
        weight_interest=0.25,
        weight_motivation=0.20,
        weight_confidence=0.25,
        weight_market=0.15,
        weight_sentiment=0.15,
        growth_rate_ceiling=20.0,
        citation_velocity_ceiling=10.0,
    ),
    "environmental": FieldProfile(
        field_category="environmental",
        display_name="Environmental / Earth Science",
        pace="moderate",
        typical_cycle_years=3.0,
        weight_interest=0.20,
        weight_motivation=0.20,
        weight_confidence=0.20,
        weight_market=0.20,
        weight_sentiment=0.20,
        growth_rate_ceiling=20.0,
        citation_velocity_ceiling=10.0,
    ),
    "social_science": FieldProfile(
        field_category="social_science",
        display_name="Social Science / Humanities",
        pace="slow",
        typical_cycle_years=5.0,
        weight_interest=0.20,
        weight_motivation=0.25,
        weight_confidence=0.20,
        weight_market=0.10,
        weight_sentiment=0.25,
        growth_rate_ceiling=15.0,
        citation_velocity_ceiling=5.0,
    ),
    "general": FieldProfile(
        field_category="general",
        display_name="General / Interdisciplinary",
        pace="moderate",
        typical_cycle_years=2.5,
        weight_interest=0.25,
        weight_motivation=0.15,
        weight_confidence=0.20,
        weight_market=0.25,
        weight_sentiment=0.15,
        growth_rate_ceiling=25.0,
        citation_velocity_ceiling=12.0,
    ),
}


def detect_field(query: str, abstracts_sample: list[str] | None = None) -> FieldProfile:
    """Detect the research field from query keywords and optional abstracts.

    Scans query first; if no match, samples abstracts for domain signals.
    Falls back to "general" profile.
    """
    combined_text = query.lower()
    if abstracts_sample:
        combined_text += " " + " ".join(a[:300].lower() for a in abstracts_sample[:20])

    for pattern, category in _FIELD_PATTERNS:
        if re.search(pattern, combined_text, re.IGNORECASE):
            return _PROFILES[category]

    return _PROFILES["general"]


def get_profile(category: str) -> FieldProfile:
    """Get a field profile by category name."""
    return _PROFILES.get(category, _PROFILES["general"])
