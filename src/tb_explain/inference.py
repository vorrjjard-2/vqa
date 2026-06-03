"""Public entry point: detector output -> natural-language explanation."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from tb_explain.schema import DetectorOutput

if TYPE_CHECKING:
    from transformers import Pipeline

MODEL_ID = "microsoft/Phi-3.5-mini-instruct"

SYSTEM_PROMPT = (
    "You are a radiology AI assistant. Given structured output from a TB detection model, "
    "write a concise clinical summary (2-4 sentences) suitable for radiologist review. "
    "Do not invent findings beyond what the JSON contains."
)


def _format_user_message(output: DetectorOutput) -> str:
    return (
        "TB detector output:\n"
        + output.model_dump_json(indent=2)
        + "\n\nSummarise these findings in plain clinical language, as concise as possible."
    )


@lru_cache(maxsize=1)
def _load_pipeline() -> "Pipeline":
    from transformers import pipeline

    return pipeline(
        "text-generation",
        model=MODEL_ID,
    )


def explain(output: DetectorOutput, max_new_tokens: int = 200) -> str:
    """Return a natural-language explanation for a DetectorOutput.

    Lazy-loads the model on first call; subsequent calls reuse the cached pipeline.
    """
    pipe = _load_pipeline()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _format_user_message(output)},
    ]
    result = pipe(messages, max_new_tokens=max_new_tokens)
    return result[0]["generated_text"][-1]["content"].strip()
