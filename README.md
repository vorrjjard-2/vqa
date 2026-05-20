# vqa

A Python module that turns tuberculosis (TB) classifier outputs into natural-language descriptions by passing structured JSON to a fine-tuned LLM.

## Motivation

Image classifiers for TB produce structured outputs — confidence scores, bounding boxes, and labels — but clinicians and downstream systems often need natural-language summaries. This project bridges that gap: classifier output → structured JSON → LLM → human-readable text.

## Pipeline

```
TB classifier  →  structured JSON  →  fine-tuned LLM  →  natural-language output
```

Each classifier prediction is serialized into a JSON object with at minimum:

- `label` — predicted class
- `confidence` — model confidence score
- `bounding_boxes` — list of detected regions (when applicable)

## Planned phases

1. **Schema definition** — settle on the JSON contract between classifier and LLM.
2. **Dataset construction** — collect/curate pairs of `(structured JSON, desired text output)` for supervised fine-tuning.
3. **Fine-tuning** — train an LLM on the JSON → text mapping.
4. **Module API** — expose a clean Python interface so a classifier's output can be fed directly in.
5. **Evaluation** — measure faithfulness, fluency, and clinical accuracy of generated text.

## Status

Early scaffolding. Schema, dataset format, and target LLM not yet decided.
