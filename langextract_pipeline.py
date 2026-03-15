import asyncio
import json
import unicodedata
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import langextract as lx
from langextract import exceptions as lx_exceptions
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


def is_transient_lx_error(exc: Exception) -> bool:
    error_text = str(exc)
    return (
        isinstance(exc, lx_exceptions.InferenceRuntimeError)
        and any(code in error_text for code in ["408", "429", "500", "502", "503", "504", "UNAVAILABLE"])
    )


def _normalize_segment_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    without_diacritics = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    return without_diacritics.casefold()


def load_examples_from_json(
    path: str | Path,
    segment: str,
) -> list[lx.data.ExampleData]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    normalized_segment = _normalize_segment_key(segment)

    matched_items = None
    for raw_segment, items in raw.items():
        if _normalize_segment_key(raw_segment) == normalized_segment:
            matched_items = items
            break

    if matched_items is None:
        available_segments = ", ".join(sorted(raw.keys()))
        raise KeyError(
            f"Segment '{segment}' was not found in examples JSON. "
            f"Available segments: {available_segments}"
        )

    return [
        lx.data.ExampleData(
            text=item["text"],
            extractions=[
                lx.data.Extraction(
                    extraction_class=extraction["extraction_class"],
                    extraction_text=extraction["extraction_text"],
                    attributes=extraction["attributes"],
                )
                for extraction in item["extractions"]
            ],
        )
        for item in matched_items
    ]


def combine_offer_documents_text(offer: dict[str, Any]) -> str:
    combined_parts: list[str] = []

    for index, document in enumerate(offer.get("documents", []), start=1):
        filename = document.get("filename") or f"document_{index}"
        document_id = document.get("id")
        ocr_text = document.get("ocr_text", "")

        if not ocr_text:
            continue

        header_lines = [f"=== DOCUMENT {index} ===", f"filename: {filename}"]
        if document_id is not None:
            header_lines.append(f"document_id: {document_id}")
        header_lines.append("ocr_text:")

        combined_parts.append("\n".join(header_lines) + f"\n{ocr_text}")

    return "\n\n".join(combined_parts)


def group_extractions_by_field_name(extractions: list[Any]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)

    for extraction in extractions:
        attributes = dict(getattr(extraction, "attributes", {}) or {})
        field_name = attributes.get("field_name")
        extraction_text = getattr(extraction, "extraction_text", None)

        if not field_name or extraction_text in (None, ""):
            continue

        extraction_value = str(extraction_text)
        if extraction_value not in grouped[field_name]:
            grouped[field_name].append(extraction_value)

    return dict(grouped)


@retry(
    reraise=True,
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception(is_transient_lx_error),
)
def extract_offer_sync(
    offer: dict[str, Any],
    prompt: str,
    examples: list[lx.data.ExampleData],
    api_key: str,
    model_id: str = "gemini-2.5-flash",
    extraction_passes: int = 1,
    max_workers: int = 2,
    batch_length: int = 6,
    max_char_buffer: int = 12000,
):
    combined_text = combine_offer_documents_text(offer)

    result = lx.extract(
        text_or_documents=combined_text,
        prompt_description=prompt,
        model_id=model_id,
        examples=examples,
        extraction_passes=extraction_passes,
        max_workers=max_workers,
        batch_length=batch_length,
        max_char_buffer=max_char_buffer,
        api_key=api_key,
    )

    extracted_values = group_extractions_by_field_name(
        getattr(result, "extractions", []),
    )

    return SimpleNamespace(
        offer_id=offer.get("id"),
        insurer=offer.get("insurer"),
        label=offer.get("label"),
        combined_text=combined_text,
        raw_result=result,
        extracted_values=extracted_values,
        extractions=getattr(result, "extractions", []),
    )


async def extract_offer_async(
    offer: dict[str, Any],
    prompt: str,
    examples: list[lx.data.ExampleData],
    api_key: str,
    model_id: str = "gemini-2.5-flash",
    extraction_passes: int = 1,
    max_workers: int = 2,
    batch_length: int = 6,
    max_char_buffer: int = 12000,
):
    return await asyncio.to_thread(
        extract_offer_sync,
        offer,
        prompt,
        examples,
        api_key,
        model_id,
        extraction_passes,
        max_workers,
        batch_length,
        max_char_buffer,
    )


async def extract_offer_values_async(
    offer: dict[str, Any],
    prompt: str,
    examples: list[lx.data.ExampleData],
    api_key: str,
    model_id: str = "gemini-2.5-flash",
    extraction_passes: int = 1,
    max_workers: int = 2,
    batch_length: int = 6,
    max_char_buffer: int = 12000,
) -> dict[str, list[str]]:
    result = await extract_offer_async(
        offer=offer,
        prompt=prompt,
        examples=examples,
        api_key=api_key,
        model_id=model_id,
        extraction_passes=extraction_passes,
        max_workers=max_workers,
        batch_length=batch_length,
        max_char_buffer=max_char_buffer,
    )
    return result.extracted_values
