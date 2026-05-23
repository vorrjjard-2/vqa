"""Bounding box -> anatomical region string.

Current vocabulary: 3x3 grid (`upper|middle|lower` x `left|center|right`).

Naive implementation: grids over the full image bounds, not a segmented lung
field. Good enough as a starting point if inputs are already lung-cropped; swap
in lung-segmentation-aware logic later.
"""

from tb_explain.schema import Location

# (x0, y0, x1, y1) in pixel space, top-left origin.
BBox = tuple[float, float, float, float]


def bbox_to_location(bbox: BBox, image_size: tuple[int, int]) -> Location:
    """Map a pixel-space bbox to one of the 9 grid cells by its center point."""
    x0, y0, x1, y1 = bbox
    width, height = image_size
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

    if cx < width / 3:
        col = "left"
    elif cx < 2 * width / 3:
        col = "center"
    else:
        col = "right"

    if cy < height / 3:
        row = "upper"
    elif cy < 2 * height / 3:
        row = "middle"
    else:
        row = "lower"

    return f"{row}_{col}"  # type: ignore[return-value]
