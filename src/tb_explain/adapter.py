"""Raw detector output -> DetectorOutput (the LLM input contract).

Responsibilities:
  1. argmax over per-class probabilities to pick `predicted_label`.
  2. Bucket raw [0, 1] detection confidences into low / medium / high bands.
  3. Map pixel-space bboxes to anatomical region strings via `anatomy`.
  4. Assemble and validate the final `DetectorOutput`.
"""

from tb_explain.anatomy import BBox, bbox_to_location
from tb_explain.schema import (
    ConfidenceBand,
    DetectorOutput,
    ImageClassification,
    PredictedLabel,
    Region,
    RegionType,
)

# Thresholds are inclusive on the lower bound: conf >= HIGH -> "high", etc.
DEFAULT_BAND_THRESHOLDS: tuple[float, float] = (0.5, 0.8)


def confidence_to_band(
    confidence: float,
    thresholds: tuple[float, float] = DEFAULT_BAND_THRESHOLDS,
) -> ConfidenceBand:
    low_cut, high_cut = thresholds
    if confidence >= high_cut:
        return "high"
    if confidence >= low_cut:
        return "medium"
    return "low"


def adapt(
    probabilities: dict[PredictedLabel, float],
    detections: list[tuple[BBox, RegionType, float]],
    image_size: tuple[int, int],
    band_thresholds: tuple[float, float] = DEFAULT_BAND_THRESHOLDS,
) -> DetectorOutput:
    """Build a validated `DetectorOutput` from raw detector outputs.

    Args:
        probabilities: per-class probabilities from the classification head.
        detections: list of `(bbox, region_type, raw_confidence)` from the
            detection head. `bbox` is `(x0, y0, x1, y1)` in pixel space.
        image_size: `(width, height)` in pixels, used by the bbox -> region mapper.
        band_thresholds: `(low_cut, high_cut)` cutoffs for confidence banding.
    """
    predicted_label = max(probabilities, key=probabilities.__getitem__)

    regions = [
        Region(
            type=region_type,
            confidence_band=confidence_to_band(confidence, band_thresholds),
            location=bbox_to_location(bbox, image_size),
        )
        for bbox, region_type, confidence in detections
    ]

    return DetectorOutput(
        image_classification=ImageClassification(
            predicted_label=predicted_label,
            probabilities=probabilities,
        ),
        regions=regions,
    )
