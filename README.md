# TB Explanation Module

Turns the output of our TB classification + detection model into a natural-language explanation of that output.

## Pipeline

```
CXR image
  → TB detector (classifier + region detector)
  → structured JSON
  → fine-tuned LLM
  → natural-language explanation
```

The LLM never sees the image — only the structured detector output. This prevents the model from hallucinating findings that aren't actually in the detector's predictions.

## Schema (draft)

LLM input:

```json
{
  "image_classification": {
    "predicted_label": "tb",
    "probabilities": {
      "healthy": 0.05,
      "sick_non_tb": 0.12,
      "tb": 0.83
    }
  },
  "regions": [
    {
      "type": "active_tb",
      "confidence_band": "high",
      "location": "upper_right"
    }
  ]
}
```

- `predicted_label` ∈ {`healthy`, `sick_non_tb`, `tb`}
- `type` ∈ {`active_tb`, `latent_tb`}
- `confidence_band` ∈ {`low`, `medium`, `high`}
- `location` is a string from a fixed anatomical vocabulary (currently a 3×3 grid over the lung field)
- `regions` is `[]` when no detections

## Plan

1. Finalize schema.
2. Build anatomical mapper (bbox → region string).
3. Construct dataset: run detector on MIMIC-CXR, pair outputs with MIMIC Findings text, filter for grounding.
4. Fine-tune a medical LLM (LoRA).
5. Wire into the API.
6. Evaluate faithfulness and readability.

## Status

Early scaffolding. No code yet.