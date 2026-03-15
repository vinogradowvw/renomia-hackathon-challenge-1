from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
from langextract import data as lx_data

from extraction_prototype import MODEL_ID, get_sort_params_async
from judge import judge_offer_documents_async
from langextract_pipeline import (
    extract_offer_async as langextract_offer_async,
    load_examples_from_json,
)
from ranking import Ranking


logger = logging.getLogger(__name__)

DEFAULT_LANGEXTRACT_EXAMPLES_PATH = Path(__file__).with_name("langextract_examples.json")


def _json_pretty(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_requested_fields(
    field_types: dict[str, str],
    fields_to_extract: list[str] | tuple[str, ...] | None,
) -> dict[str, str]:
    if not fields_to_extract:
        return dict(field_types)

    missing_fields = [field_name for field_name in fields_to_extract if field_name not in field_types]
    if missing_fields:
        raise ValueError(
            "fields_to_extract contains fields missing in field_types: "
            + ", ".join(missing_fields)
        )

    return {
        field_name: field_types[field_name]
        for field_name in fields_to_extract
    }


def build_langextract_prompt(
    segment: str,
    requested_fields: dict[str, str],
) -> str:
    return f"""
You extract insurance field candidates from Czech OCR text for segment "{segment}".

Return extractions, not a final normalized JSON.

Each extraction must use:
- extraction_class = "insurance_field"
- attributes.field_name = one of the requested field names
- attributes.value_type = the loose field type from requested_fields

Rules:
1. Extract only values that are explicitly supported by the source text.
2. extraction_text must be a strict span copied from the text, preferably only the value and not the whole label line.
3. For number-like fields, extraction_text must still be the original value span from the text, not a normalized integer.
4. You may return multiple candidates for the same field if the document contains multiple relevant mentions.
5. If the field name contains "II", it usually refers to the second relevant occurrence or second variant in the document.
6. Do not invent values that do not appear in the text.
7. Prefer high-precision spans over long fragments.

REQUESTED_FIELDS:
{_json_pretty(requested_fields)}
""".strip()


def filter_langextract_examples(
    examples: list[lx_data.ExampleData],
    requested_fields: dict[str, str],
) -> list[lx_data.ExampleData]:
    requested_field_names = set(requested_fields)
    filtered_examples: list[lx_data.ExampleData] = []

    for example in examples:
        filtered_extractions = []

        for extraction in getattr(example, "extractions", []):
            attributes = dict(getattr(extraction, "attributes", {}) or {})
            field_name = attributes.get("field_name")
            if field_name not in requested_field_names:
                continue

            filtered_extractions.append(
                lx_data.Extraction(
                    extraction_class=getattr(extraction, "extraction_class", "insurance_field"),
                    extraction_text=getattr(extraction, "extraction_text", ""),
                    attributes=attributes,
                )
            )

        if not filtered_extractions:
            continue

        filtered_examples.append(
            lx_data.ExampleData(
                text=getattr(example, "text", ""),
                extractions=filtered_extractions,
            )
        )

    return filtered_examples


def load_judge_few_shot_examples(
    path: str | Path,
    segment: str,
    requested_fields: dict[str, str],
) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = raw.get(segment, [])
    few_shot_examples: list[dict[str, Any]] = []

    for item in items:
        example_return: dict[str, Any] = {}

        for extraction in item.get("extractions", []):
            attributes = extraction.get("attributes", {}) or {}
            field_name = attributes.get("field_name")
            if field_name not in requested_fields:
                continue
            if field_name in example_return:
                continue
            example_return[field_name] = extraction.get("extraction_text")

        if not example_return:
            continue

        few_shot_examples.append(
            {
                "text": item.get("text", ""),
                "requested_fields": {
                    field_name: requested_fields[field_name]
                    for field_name in example_return
                },
                "example_return": example_return,
            }
        )

    return few_shot_examples


def build_error_offer_result(
    offer: dict[str, Any],
    requested_fields: dict[str, str],
    error: Exception,
) -> SimpleNamespace:
    parsed_json = {field_name: None for field_name in requested_fields}
    final_offer = {
        "id": offer.get("id"),
        "insurer": offer.get("insurer"),
        "label": offer.get("label"),
        **parsed_json,
    }

    return SimpleNamespace(
        offer_id=offer.get("id"),
        insurer=offer.get("insurer"),
        label=offer.get("label"),
        parsed_json=parsed_json,
        final_offer=final_offer,
        langextract_values={},
        langextract_result=None,
        judge_result=None,
        errors=[
            {
                "offer_id": offer.get("id"),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        ],
    )


async def process_offer_async(
    offer: dict[str, Any],
    api_key: str,
    segment: str,
    requested_fields: dict[str, str],
    langextract_examples: list[lx_data.ExampleData],
    judge_few_shot_examples: list[dict[str, Any]] | None = None,
    model_id: str = MODEL_ID,
    langextract_extraction_passes: int = 1,
    langextract_max_workers: int = 2,
    langextract_batch_length: int = 6,
    langextract_max_char_buffer: int = 12000,
) -> SimpleNamespace:
    langextract_prompt = build_langextract_prompt(
        segment=segment,
        requested_fields=requested_fields,
    )

    logger.info("Starting full pipeline for offer_id=%s", offer.get("id"))

    langextract_result = await langextract_offer_async(
        offer=offer,
        prompt=langextract_prompt,
        examples=langextract_examples,
        api_key=api_key,
        model_id=model_id,
        extraction_passes=langextract_extraction_passes,
        max_workers=langextract_max_workers,
        batch_length=langextract_batch_length,
        max_char_buffer=langextract_max_char_buffer,
    )

    judge_result = await judge_offer_documents_async(
        offer=offer,
        langextract_values=langextract_result.extracted_values,
        api_key=api_key,
        requested_fields=requested_fields,
        few_shot_examples=judge_few_shot_examples,
        model_id=model_id,
    )

    final_offer = {
        "id": offer.get("id"),
        "insurer": offer.get("insurer"),
        "label": offer.get("label"),
        **judge_result.parsed_json,
    }

    logger.info(
        "Finished full pipeline for offer_id=%s with %s non-null fields",
        offer.get("id"),
        sum(1 for value in judge_result.parsed_json.values() if value is not None),
    )

    return SimpleNamespace(
        offer_id=offer.get("id"),
        insurer=offer.get("insurer"),
        label=offer.get("label"),
        parsed_json=judge_result.parsed_json,
        final_offer=final_offer,
        langextract_values=langextract_result.extracted_values,
        langextract_result=langextract_result,
        judge_result=judge_result,
        errors=[],
    )


async def run_pipeline_async(
    input_data: dict[str, Any],
    api_key: str,
    concurrency: int = 3,
    langextract_examples_path: str | Path = DEFAULT_LANGEXTRACT_EXAMPLES_PATH,
    model_id: str = MODEL_ID,
    include_debug_payload: bool = True,
) -> dict[str, Any]:
    offers = input_data.get("offers", [])
    segment = input_data.get("segment")
    field_types = input_data.get("field_types", {})
    fields_to_extract = input_data.get("fields_to_extract")

    if not isinstance(offers, list):
        raise ValueError("input_data['offers'] must be a list.")
    if not isinstance(field_types, dict) or not field_types:
        raise ValueError("input_data['field_types'] must be a non-empty dict.")
    if not segment:
        raise ValueError("input_data['segment'] must be provided.")

    requested_fields = build_requested_fields(
        field_types=field_types,
        fields_to_extract=fields_to_extract,
    )

    raw_examples = load_examples_from_json(
        path=langextract_examples_path,
        segment=segment,
    )
    langextract_examples = filter_langextract_examples(
        examples=raw_examples,
        requested_fields=requested_fields,
    )
    judge_few_shot_examples = load_judge_few_shot_examples(
        path=langextract_examples_path,
        segment=segment,
        requested_fields=requested_fields,
    )

    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(single_offer: dict[str, Any]) -> SimpleNamespace:
        async with semaphore:
            try:
                return await process_offer_async(
                    offer=single_offer,
                    api_key=api_key,
                    segment=segment,
                    requested_fields=requested_fields,
                    langextract_examples=langextract_examples,
                    judge_few_shot_examples=judge_few_shot_examples,
                    model_id=model_id,
                )
            except Exception as exc:
                logger.error(
                    "Full pipeline failed for offer_id=%s insurer=%s label=%s error_type=%s error=%s",
                    single_offer.get("id"),
                    single_offer.get("insurer"),
                    single_offer.get("label"),
                    type(exc).__name__,
                    str(exc),
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
                return build_error_offer_result(
                    offer=single_offer,
                    requested_fields=requested_fields,
                    error=exc,
                )

    sort_task = asyncio.create_task(
        get_sort_params_async(
            api_key=api_key,
            requested_fields=requested_fields,
            model_id=model_id,
        )
    )

    offer_results = await asyncio.gather(*(run_one(offer) for offer in offers))
    sort_info = await sort_task

    offers_parsed = [result.final_offer for result in offer_results]

    ranking_input = pd.DataFrame(offers_parsed)
    ranking = Ranking().rank(
        df=ranking_input,
        sort_params=sort_info.sort_params,
    )
    best_offer_id = ranking[0] if ranking else None

    result = {
        "segment": segment,
        "requested_fields": requested_fields,
        "fields_to_extract": list(requested_fields.keys()),
        "offers_parsed": offers_parsed,
        "ranking": ranking,
        "best_offer_id": best_offer_id,
        "sort_params": sort_info.sort_params,
    }

    if include_debug_payload:
        result["offer_results"] = [
            {
                "offer_id": item.offer_id,
                "insurer": item.insurer,
                "label": item.label,
                "langextract_values": item.langextract_values,
                "parsed_json": item.parsed_json,
                "errors": item.errors,
            }
            for item in offer_results
        ]
        result["langextract_prompt"] = build_langextract_prompt(
            segment=segment,
            requested_fields=requested_fields,
        )

    return result


def run_pipeline(
    input_data: dict[str, Any],
    api_key: str,
    concurrency: int = 3,
    langextract_examples_path: str | Path = DEFAULT_LANGEXTRACT_EXAMPLES_PATH,
    model_id: str = MODEL_ID,
    include_debug_payload: bool = True,
) -> dict[str, Any]:
    return asyncio.run(
        run_pipeline_async(
            input_data=input_data,
            api_key=api_key,
            concurrency=concurrency,
            langextract_examples_path=langextract_examples_path,
            model_id=model_id,
            include_debug_payload=include_debug_payload,
        )
    )
