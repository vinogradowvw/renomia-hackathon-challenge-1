from __future__ import annotations

import asyncio
import json
import unicodedata
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from caching.CacheRepo import CacheRepo
from caching.hashing import Hashing
from extraction_prototype import (
    DEFAULT_FIELD_TYPES,
    MODEL_ID,
    _json_pretty,
    build_offer_inputs,
    build_output_offer_dict,
    build_response_json_schema,
    coerce_parsed_json,
    extract_offer_async,
    format_document_manifest,
    get_sort_params_async,
    infer_numericish_fields,
    is_retryable,
    logger,
    normalize_number_typed_fields,
    usage_to_dict,
)
from models import Cache, transaction


DEFAULT_EXAMPLES_PATH = Path(__file__).with_name("examples.json")
CACHE_NAMESPACE = "gemini_extraction_judge_v1"
DEFAULT_JUDGE_INSTRUCTION = (
    "Check RESULT against TEXT. Keep values that are explicitly supported by the text, "
    "fix values that are wrong or over-normalized, shorten long text fields when needed, "
    "normalize number fields to plain numeric values, and set unsupported fields to null."
)


def _normalize_segment_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    without_diacritics = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    return without_diacritics.casefold()


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


def build_offer_cache_key(
    offer: dict[str, Any],
    segment: str,
    requested_fields: dict[str, str],
    extraction_few_shot_examples: list[dict[str, Any]],
    duplicated_judge_examples_prompt: str,
    model_id: str,
) -> str:
    payload = {
        "cache_namespace": CACHE_NAMESPACE,
        "segment": segment,
        "model_id": model_id,
        "requested_fields": requested_fields,
        "documents": [
            {
                "id": document.get("id"),
                "filename": document.get("filename"),
                "ocr_text": document.get("ocr_text"),
                "pdf_url": document.get("pdf_url"),
            }
            for document in offer.get("documents", [])
        ],
        "extraction_few_shot_examples": extraction_few_shot_examples,
        "duplicated_judge_examples_prompt": duplicated_judge_examples_prompt,
    }
    return f"{CACHE_NAMESPACE}:{Hashing.sha256(_json_pretty(payload))}"


def load_cached_offer_result(cache_key: str) -> SimpleNamespace | None:
    try:
        with transaction():
            cache_entry = CacheRepo.get_instance().get_by_key(cache_key)
    except Exception as exc:
        logger.warning("Cache read failed for key=%s: %s", cache_key, str(exc))
        return None

    if cache_entry is None:
        return None

    payload = cache_entry.value or {}
    return SimpleNamespace(
        offer_id=payload.get("offer_id"),
        insurer=payload.get("insurer"),
        label=payload.get("label"),
        parsed_json=payload.get("parsed_json", {}),
        final_offer=payload.get("final_offer", {}),
        extraction_result=None,
        judge_result=None,
        errors=[],
        cache_hit=True,
        cache_key=cache_key,
    )


def save_cached_offer_result(
    cache_key: str,
    offer_result: SimpleNamespace,
) -> bool:
    cache_value = {
        "offer_id": offer_result.offer_id,
        "insurer": offer_result.insurer,
        "label": offer_result.label,
        "parsed_json": offer_result.parsed_json,
        "final_offer": offer_result.final_offer,
    }

    try:
        with transaction():
            was_saved = CacheRepo.get_instance().add(
                Cache(
                    key=cache_key,
                    value=cache_value,
                )
            )
        if was_saved:
            logger.info(
                "Saved parsed offer to cache for offer_id=%s key=%s",
                offer_result.offer_id,
                cache_key,
            )
        else:
            logger.info(
                "Parsed offer was already present in cache for offer_id=%s key=%s",
                offer_result.offer_id,
                cache_key,
            )
        return was_saved
    except Exception as exc:
        logger.warning("Cache write failed for key=%s: %s", cache_key, str(exc))
        return False


def load_static_examples(
    path: str | Path,
    segment: str,
) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    normalized_segment = _normalize_segment_key(segment)

    matched_examples = [
        item
        for item in raw.get("examples", [])
        if _normalize_segment_key(item.get("segment", "")) == normalized_segment
    ]

    if not matched_examples:
        available_segments = sorted(
            {
                item.get("segment", "")
                for item in raw.get("examples", [])
                if item.get("segment")
            }
        )
        raise KeyError(
            f"Segment '{segment}' was not found in examples.json. "
            f"Available segments: {', '.join(available_segments)}"
        )

    return matched_examples


def build_extraction_few_shot_examples(
    static_examples: list[dict[str, Any]],
    requested_fields: dict[str, str],
) -> list[dict[str, Any]]:
    few_shot_examples: list[dict[str, Any]] = []

    for item in static_examples:
        requested_subset = {
            field_name: requested_fields[field_name]
            for field_name in item.get("requested_fields", {})
            if field_name in requested_fields
        }
        example_return_subset = {
            field_name: value
            for field_name, value in item.get("example_return", {}).items()
            if field_name in requested_subset
        }

        if not requested_subset:
            continue

        few_shot_examples.append(
            {
                "text": item.get("text", ""),
                "requested_fields": requested_subset,
                "example_return": example_return_subset,
            }
        )

    return few_shot_examples


def build_duplicated_judge_examples_prompt(
    static_examples: list[dict[str, Any]],
    requested_fields: dict[str, str],
) -> str:
    filtered_examples = build_extraction_few_shot_examples(
        static_examples=static_examples,
        requested_fields=requested_fields,
    )

    if not filtered_examples:
        return "TEXT:\n\nRESULT:\n{}\nJUDGE INSTRUCTION:\n" + DEFAULT_JUDGE_INSTRUCTION

    if len(filtered_examples) == 1:
        selected_examples = [filtered_examples[0], filtered_examples[0]]
    else:
        selected_examples = filtered_examples[:2]

    blocks: list[str] = []
    for example in selected_examples:
        blocks.append("TEXT:")
        blocks.append(example["text"])
        blocks.append("")
        blocks.append("RESULT:")
        blocks.append(_json_pretty(example["example_return"]))
        blocks.append("")
        blocks.append("JUDGE INSTRUCTION:")
        blocks.append(DEFAULT_JUDGE_INSTRUCTION)
        blocks.append("")

    return "\n".join(blocks).strip()


def build_gemini_judge_prompt(
    requested_fields: dict[str, str],
    combined_ocr_text: str,
    source_documents: list[dict[str, Any]],
    preliminary_result: dict[str, Any],
    duplicated_judge_examples_prompt: str,
) -> str:
    numericish_fields = infer_numericish_fields(requested_fields)
    numericish_block = "\n".join(f"- {field_name}" for field_name in sorted(numericish_fields))

    if not numericish_block:
        numericish_block = "- none"

    return f"""
CURRENT_TEXT:
{combined_ocr_text}

CURRENT_RESULT:
{_json_pretty(preliminary_result)}

Requested fields:
{_json_pretty(requested_fields)}

Numeric-like fields:
{numericish_block}

{duplicated_judge_examples_prompt}


CURRENT_JUDGE_INSTRUCTION:
You are validating a preliminary Gemini extraction for Czech insurance documents.

Your task is to verify the preliminary extraction result against the actual source text and attached PDF files.

Check CURRENT_RESULT against CURRENT_TEXT and the attached documents. Fix mistakes, preserve supported values, normalize number fields, shorten long text fields when needed, and return null for unsupported or unclear fields.

Rules:
1. Keep exactly the same keys as in requested_fields.
2. The preliminary result is only a draft. You may keep, fix or replace values.
3. Do not invent values that are not supported by the documents.
4. For number-typed fields, return only the numeric value when possible.
5. For text fields, if the correct value would be too long, shorten it into a concise but faithful summary.
6. If a field remains unclear after checking the documents, return null.
7. If the field name contains "II", it usually refers to the second relevant occurrence or second variant in the document.
8. Return only valid JSON.


CURRENT_DOCUMENT_MANIFEST:
{format_document_manifest(source_documents)}
""".strip()


@retry(
    reraise=True,
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception(is_retryable),
)
async def judge_extraction_async(
    client,
    offer: dict[str, Any],
    preliminary_result: dict[str, Any],
    requested_fields: dict[str, str],
    duplicated_judge_examples_prompt: str,
    model_id: str = MODEL_ID,
):
    offer_id = offer.get("id")
    logger.info("Starting Gemini judge for offer_id=%s", offer_id)

    combined_ocr_text, source_documents, pdf_files = await build_offer_inputs(
        client=client,
        offer=offer,
    )
    response_json_schema = build_response_json_schema(requested_fields)
    prompt = build_gemini_judge_prompt(
        requested_fields=requested_fields,
        combined_ocr_text=combined_ocr_text,
        source_documents=source_documents,
        preliminary_result=preliminary_result,
        duplicated_judge_examples_prompt=duplicated_judge_examples_prompt,
    )
    prompt = prompt + "\n" + prompt
    model_contents = [prompt, *pdf_files]
    generate_config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
        response_json_schema=response_json_schema,
    )

    estimated_input_tokens = None
    try:
        token_count_resp = await client.models.count_tokens(
            model=model_id,
            contents=model_contents,
        )
        estimated_input_tokens = getattr(token_count_resp, "total_tokens", None)
    except Exception:
        token_count_resp = None

    try:
        response = await client.models.generate_content(
            model=model_id,
            contents=model_contents,
            config=generate_config,
        )
    except Exception as exc:
        if pdf_files and "The document has no pages." in str(exc):
            logger.warning(
                "Falling back to OCR-only Gemini judge for offer_id=%s because attached file processing failed: %s",
                offer_id,
                str(exc),
            )
            response = await client.models.generate_content(
                model=model_id,
                contents=[prompt],
                config=generate_config,
            )
        else:
            raise

    raw_text = response.text or "{}"
    parsed_json = coerce_parsed_json(json.loads(raw_text), requested_fields)
    parsed_json = normalize_number_typed_fields(parsed_json, requested_fields)

    logger.info(
        "Gemini judge finished for offer_id=%s with %s non-null fields",
        offer_id,
        sum(1 for value in parsed_json.values() if value is not None),
    )

    return SimpleNamespace(
        offer_id=offer.get("id"),
        insurer=offer.get("insurer"),
        label=offer.get("label"),
        parsed_json=parsed_json,
        prompt=prompt,
        combined_ocr_text=combined_ocr_text,
        source_documents=source_documents,
        attached_pdf_files=pdf_files,
        preliminary_result=preliminary_result,
        estimated_input_tokens=estimated_input_tokens,
        usage=usage_to_dict(getattr(response, "usage_metadata", None)),
        raw_response_text=raw_text,
    )


def build_error_offer_result(
    offer: dict[str, Any],
    requested_fields: dict[str, str],
    error: Exception,
) -> SimpleNamespace:
    parsed_json = {field_name: None for field_name in requested_fields}
    final_offer = build_output_offer_dict(
        offer=offer,
        parsed_json=parsed_json,
        requested_fields=requested_fields,
    )

    return SimpleNamespace(
        offer_id=offer.get("id"),
        insurer=offer.get("insurer"),
        label=offer.get("label"),
        parsed_json=parsed_json,
        final_offer=final_offer,
        extraction_result=None,
        judge_result=None,
        errors=[
            {
                "offer_id": offer.get("id"),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        ],
        cache_hit=False,
        cache_key=None,
        cache_saved=False,
    )


async def process_offer_async(
    client,
    offer: dict[str, Any],
    requested_fields: dict[str, str],
    extraction_few_shot_examples: list[dict[str, Any]],
    duplicated_judge_examples_prompt: str,
    model_id: str = MODEL_ID,
) -> SimpleNamespace:
    logger.info("Starting Gemini extraction+judge pipeline for offer_id=%s", offer.get("id"))

    extraction_result = await extract_offer_async(
        client=client,
        offer=offer,
        requested_fields=requested_fields,
        few_shot_examples=extraction_few_shot_examples,
        model_id=model_id,
    )

    judge_result = await judge_extraction_async(
        client=client,
        offer=offer,
        preliminary_result=extraction_result.parsed_json,
        requested_fields=requested_fields,
        duplicated_judge_examples_prompt=duplicated_judge_examples_prompt,
        model_id=model_id,
    )

    final_offer = build_output_offer_dict(
        offer=offer,
        parsed_json=judge_result.parsed_json,
        requested_fields=requested_fields,
    )

    logger.info(
        "Finished Gemini extraction+judge pipeline for offer_id=%s with %s non-null fields",
        offer.get("id"),
        sum(1 for value in judge_result.parsed_json.values() if value is not None),
    )

    return SimpleNamespace(
        offer_id=offer.get("id"),
        insurer=offer.get("insurer"),
        label=offer.get("label"),
        parsed_json=judge_result.parsed_json,
        final_offer=final_offer,
        extraction_result=extraction_result,
        judge_result=judge_result,
        errors=[],
        cache_hit=False,
        cache_key=None,
        cache_saved=False,
    )


def build_ranking_result(
    offers_parsed: list[dict[str, Any]],
    sort_params: list[dict[str, Any]],
    null_threshold: float = 0.67,
) -> tuple[list[str], str | None]:
    try:
        import pandas as pd
        from ranking import Ranking
    except Exception as exc:
        logger.warning(
            "Ranking dependencies are unavailable, falling back to input order: %s",
            str(exc),
        )
        ranking = [str(item["id"]) for item in offers_parsed]
        return ranking, ranking[0] if ranking else None

    df = pd.DataFrame(offers_parsed)
    ranking = Ranking().rank(
        df=df,
        sort_params=sort_params,
        null_threshold=null_threshold,
    )
    best_offer_id = ranking[0] if ranking else None
    return ranking, best_offer_id


async def pipeline(
    input_data: dict[str, Any],
    api_key: str,
    concurrency: int = 3,
    examples_path: str | Path = DEFAULT_EXAMPLES_PATH,
    model_id: str = MODEL_ID,
    include_debug_payload: bool = True,
    use_cache: bool = True,
    null_threshold: float = 0.67,
) -> dict[str, Any]:
    offers = input_data.get("offers", [])
    segment = input_data.get("segment")
    field_types = input_data.get("field_types", {})
    fields_to_extract = input_data.get("fields_to_extract")

    if not isinstance(offers, list):
        raise ValueError("input_data['offers'] must be a list.")
    if not isinstance(field_types, dict) or not field_types:
        requested_fields = dict(DEFAULT_FIELD_TYPES)
    else:
        requested_fields = build_requested_fields(
            field_types=field_types,
            fields_to_extract=fields_to_extract,
        )
    if not segment:
        raise ValueError("input_data['segment'] must be provided.")

    static_examples = load_static_examples(
        path=examples_path,
        segment=segment,
    )
    extraction_few_shot_examples = build_extraction_few_shot_examples(
        static_examples=static_examples,
        requested_fields=requested_fields,
    )
    duplicated_judge_examples_prompt = build_duplicated_judge_examples_prompt(
        static_examples=static_examples,
        requested_fields=requested_fields,
    )

    semaphore = asyncio.Semaphore(concurrency)
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=120_000,
        ),
    ).aio

    async def run_one(single_offer: dict[str, Any]) -> SimpleNamespace:
        async with semaphore:
            cache_key = None
            if use_cache:
                cache_key = build_offer_cache_key(
                    offer=single_offer,
                    segment=segment,
                    requested_fields=requested_fields,
                    extraction_few_shot_examples=extraction_few_shot_examples,
                    duplicated_judge_examples_prompt=duplicated_judge_examples_prompt,
                    model_id=model_id,
                )
                cached_result = load_cached_offer_result(cache_key)
                if cached_result is not None:
                    logger.info(
                        "Cache hit for offer_id=%s key=%s",
                        single_offer.get("id"),
                        cache_key,
                    )
                    return cached_result

            try:
                result = await process_offer_async(
                    client=client,
                    offer=single_offer,
                    requested_fields=requested_fields,
                    extraction_few_shot_examples=extraction_few_shot_examples,
                    duplicated_judge_examples_prompt=duplicated_judge_examples_prompt,
                    model_id=model_id,
                )
                result.cache_key = cache_key
                if use_cache and cache_key is not None:
                    result.cache_saved = save_cached_offer_result(
                        cache_key=cache_key,
                        offer_result=result,
                    )
                return result
            except Exception as exc:
                logger.error(
                    "Gemini extraction+judge pipeline failed for offer_id=%s insurer=%s label=%s error_type=%s error=%s",
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

    try:
        offer_results = await asyncio.gather(*(run_one(offer) for offer in offers))
    finally:
        await client.aclose()

    sort_info = await sort_task

    offers_parsed = [result.final_offer for result in offer_results]
    ranking, best_offer_id = build_ranking_result(
        offers_parsed=offers_parsed,
        sort_params=sort_info.sort_params,
        null_threshold=null_threshold,
    )
    formatted_offers_parsed = [
        build_output_offer_dict(
            offer={
                "id": item.offer_id,
                "insurer": item.insurer,
                "label": item.label,
            },
            parsed_json=item.parsed_json,
            requested_fields=requested_fields,
            none_placeholder="N/A",
        )
        for item in offer_results
    ]

    result = {
        "segment": segment,
        "requested_fields": requested_fields,
        "fields_to_extract": list(requested_fields.keys()),
        "offers_parsed": formatted_offers_parsed,
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
                "parsed_json": item.parsed_json,
                "errors": item.errors,
                "cache_hit": getattr(item, "cache_hit", False),
                "cache_key": getattr(item, "cache_key", None),
                "cache_saved": getattr(item, "cache_saved", False),
                "extraction_raw_response_text": None
                if item.extraction_result is None
                else item.extraction_result.raw_response_text,
                "judge_raw_response_text": None
                if item.judge_result is None
                else item.judge_result.raw_response_text,
            }
            for item in offer_results
        ]
        result["judge_prompt_examples"] = duplicated_judge_examples_prompt

    return result


def run_pipeline_sync(
    input_data: dict[str, Any],
    api_key: str,
    concurrency: int = 3,
    examples_path: str | Path = DEFAULT_EXAMPLES_PATH,
    model_id: str = MODEL_ID,
    include_debug_payload: bool = True,
    use_cache: bool = True,
    null_threshold: float = 0.67,
) -> dict[str, Any]:
    return asyncio.run(
        pipeline(
            input_data=input_data,
            api_key=api_key,
            concurrency=concurrency,
            examples_path=examples_path,
            model_id=model_id,
            include_debug_payload=include_debug_payload,
            use_cache=use_cache,
            null_threshold=null_threshold,
        )
    )
