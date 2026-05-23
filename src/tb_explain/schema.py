"""Pydantic schema for the JSON payload passed from detector to LLM.

Mirrors the schema draft in README.md. Edit both together.
"""

from typing import Literal

from pydantic import BaseModel

PredictedLabel = Literal["healthy", "sick_non_tb", "tb"]
RegionType = Literal["active_tb", "latent_tb"]
ConfidenceBand = Literal["low", "medium", "high"]
Location = Literal[
    "upper_left", "upper_center", "upper_right",
    "middle_left", "middle_center", "middle_right",
    "lower_left", "lower_center", "lower_right",
]


class ImageClassification(BaseModel):
    predicted_label: PredictedLabel
    probabilities: dict[PredictedLabel, float]


class Region(BaseModel):
    type: RegionType
    confidence_band: ConfidenceBand
    location: Location


class DetectorOutput(BaseModel):
    """Top-level JSON contract: detector output, LLM input."""

    image_classification: ImageClassification
    regions: list[Region] = []
