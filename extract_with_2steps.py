import json
from types import SimpleNamespace
from typing import Any

from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from extraction_prototype import (
    DEFAULT_FIELD_TYPES,
    MODEL_ID,
    _json_pretty,
    build_offer_inputs,
    format_document_manifest,
    format_few_shot_examples,
    infer_numericish_fields,
    is_retryable,
    make_extractions_from_json,
    usage_to_dict,
)


def build_candidate_response_json_schema(requested_fields: dict[str, str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field_name": {
                            "type": "string",
                            "enum": list(requested_fields.keys()),
                        },
                        "raw_value_text": {
                            "type": ["string", "null"],
                        },
                        "evidence_text": {
                            "type": "string",
                        },
                        "line_start": {
                            "type": "integer",
                        },
                        "line_end": {
                            "type": "integer",
                        },
                        "why_candidate": {
                            "type": ["string", "null"],
                        },
                    },
                    "required": [
                        "field_name",
                        "raw_value_text",
                        "evidence_text",
                        "line_start",
                        "line_end",
                        "why_candidate",
                    ],
                },
            },
        },
        "required": ["candidates"],
    }


def build_final_response_json_schema(requested_fields: dict[str, str]) -> dict[str, Any]:
    properties = {}

    for field_name, loose_type in requested_fields.items():
        normalized_type = str(loose_type).strip().lower()

        if normalized_type in {"int", "integer"}:
            schema_type = ["integer", "null"]
        elif normalized_type in {"number", "float", "numeric", "decimal"}:
            schema_type = ["integer", "number", "null"]
        elif normalized_type in {"bool", "boolean"}:
            schema_type = ["boolean", "null"]
        elif normalized_type in {"array", "list"}:
            schema_type = ["array", "null"]
        elif normalized_type in {"object", "dict"}:
            schema_type = ["object", "null"]
        else:
            schema_type = ["string", "null"]

        properties[field_name] = {
            "type": schema_type,
            "description": (
                f"Final resolved value for field '{field_name}'. "
                f"Requested loose type: {loose_type}."
            ),
        }

    return {
        "type": "object",
        "properties": properties,
        "required": list(requested_fields.keys()),
    }


def build_numbered_ocr_text(combined_ocr_text: str) -> tuple[str, list[str]]:
    raw_lines = combined_ocr_text.splitlines()
    numbered_lines = [
        f"[line {line_number:04d}] {line}"
        for line_number, line in enumerate(raw_lines, start=1)
    ]
    return "\n".join(numbered_lines), raw_lines


def coerce_candidate_response(parsed_json: Any, requested_fields: dict[str, str]) -> list[dict[str, Any]]:
    if isinstance(parsed_json, list):
        raw_candidates = parsed_json
    elif isinstance(parsed_json, dict):
        raw_candidates = parsed_json.get("candidates", [])
    else:
        raise ValueError("Candidate response is neither a JSON object nor a JSON array.")

    candidates = []
    valid_fields = set(requested_fields.keys())

    for item in raw_candidates:
        if not isinstance(item, dict):
            continue

        field_name = item.get("field_name")
        if field_name not in valid_fields:
            continue

        try:
            line_start = int(item.get("line_start", 0))
            line_end = int(item.get("line_end", 0))
        except (TypeError, ValueError):
            line_start = 0
            line_end = 0

        candidates.append(
            {
                "field_name": field_name,
                "raw_value_text": item.get("raw_value_text"),
                "evidence_text": item.get("evidence_text", ""),
                "line_start": line_start,
                "line_end": line_end,
                "why_candidate": item.get("why_candidate"),
            }
        )

    return candidates


def coerce_final_response(parsed_json: Any, requested_fields: dict[str, str]) -> dict[str, Any]:
    if not isinstance(parsed_json, dict):
        raise ValueError("Final model response is not a JSON object.")

    return {
        field_name: parsed_json.get(field_name)
        for field_name in requested_fields
    }


def build_candidate_contexts(
    candidates: list[dict[str, Any]],
    raw_lines: list[str],
    radius: int = 2,
) -> list[dict[str, Any]]:
    contexts = []

    for index, candidate in enumerate(candidates, start=1):
        line_start = candidate["line_start"]
        line_end = candidate["line_end"]

        if not raw_lines or line_start <= 0 or line_end <= 0:
            contexts.append(
                {
                    "candidate_index": index,
                    "field_name": candidate["field_name"],
                    "line_start": line_start,
                    "line_end": line_end,
                    "context_text": None,
                }
            )
            continue

        safe_start = max(1, line_start - radius)
        safe_end = min(len(raw_lines), line_end + radius)

        snippet_lines = [
            f"[line {line_number:04d}] {raw_lines[line_number - 1]}"
            for line_number in range(safe_start, safe_end + 1)
        ]

        contexts.append(
            {
                "candidate_index": index,
                "field_name": candidate["field_name"],
                "line_start": line_start,
                "line_end": line_end,
                "context_text": "\n".join(snippet_lines),
            }
        )

    return contexts


def build_evidence_prompt(
    requested_fields: dict[str, str],
    source_documents: list[dict[str, Any]],
    numbered_ocr_text: str,
) -> str:
    numericish_fields = infer_numericish_fields(requested_fields)
    numericish_block = "\n".join(f"- {field_name}" for field_name in sorted(numericish_fields))

    if not numericish_block:
        numericish_block = "- none"

    return f"""
You are extracting candidate evidence for insurance fields from offer documents.

Task:
Find possible text fragments that may contain values for the requested fields.

Input handling:
- Some documents may be attached as real PDF files when `pdf_url` is available.
- Documents without `pdf_url` are provided below as line-numbered OCR text.
- Use both attached PDFs and OCR text together as the source.
- If a candidate is supported by OCR text, provide real line_start and line_end values from the numbered OCR text.
- If a candidate comes only from an attached PDF without OCR line mapping, use 0 for line_start and line_end.

Rules:
1. Do not return final normalized values yet.
2. Return candidates only when there is explicit textual support.
3. A candidate may still be useful even if the value is a few lines away from the field label.
4. Use nearby context and insurance wording to associate values with fields.
5. If unsure, prefer returning multiple candidates rather than guessing one final answer.
6. Copy evidence_text as faithfully as possible from the source.
7. Include line_start and line_end.
8. If the field name contains "II", it usually refers to the second relevant occurrence or the second variant of that value in the documents.
9. For numeric-like fields, candidates should usually contain digits.

Requested fields:
{_json_pretty(requested_fields)}

Numeric-like fields:
{numericish_block}

CURRENT_DOCUMENT_MANIFEST:
{format_document_manifest(source_documents)}

CURRENT_LINE_NUMBERED_OCR_TEXT_FOR_DOCUMENTS_WITHOUT_PDF_URL:
{numbered_ocr_text}
""".strip()


def build_resolution_prompt(
    requested_fields: dict[str, str],
    candidates: list[dict[str, Any]],
    candidate_contexts: list[dict[str, Any]],
    few_shot_examples: list[Any] | None = None,
) -> str:
    numericish_fields = infer_numericish_fields(requested_fields)
    numericish_block = "\n".join(f"- {field_name}" for field_name in sorted(numericish_fields))

    if not numericish_block:
        numericish_block = "- none"

    return f"""
You are resolving candidate evidence into a final structured extraction.

Task:
For each target field, choose the best candidate based only on the provided candidates and local context.

Rules:
1. Prefer candidates explicitly tied to the field label.
2. Distinguish main limits from sublimits and do not confuse nearby but different fields.
3. For numeric-like fields, normalize Czech monetary or count values into plain integer numbers whenever the value is clear.
4. If multiple candidates conflict and the correct one is unclear, return null.
5. Do not invent values.
6. Use only the supplied candidates and supplied local context.
7. If the field name contains "II", prefer the second relevant occurrence or second variant if the evidence supports that interpretation.
8. If text fields would otherwise be too long, shorten them into a concise but faithful summary.

Requested fields:
{_json_pretty(requested_fields)}

Numeric-like fields:
{numericish_block}

{format_few_shot_examples(few_shot_examples)}

EVIDENCE_CANDIDATES:
{_json_pretty(candidates)}

LOCAL_CONTEXT_SNIPPETS:
{_json_pretty(candidate_contexts)}
""".strip()


@retry(
    reraise=True,
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
)
async def extract_offer_async(
    client,
    offer: dict[str, Any],
    requested_fields: dict[str, str],
    few_shot_examples: list[Any] | None = None,
    model_id: str = MODEL_ID,
):
    combined_ocr_text, source_documents, pdf_files = await build_offer_inputs(
        client=client,
        offer=offer,
    )
    numbered_ocr_text, raw_lines = build_numbered_ocr_text(combined_ocr_text)

    evidence_prompt = build_evidence_prompt(
        requested_fields=requested_fields,
        source_documents=source_documents,
        numbered_ocr_text=numbered_ocr_text,
    )
    evidence_contents = [evidence_prompt, *pdf_files]

    evidence_estimated_input_tokens = None
    try:
        token_count_resp_1 = await client.models.count_tokens(
            model=model_id,
            contents=evidence_contents,
        )
        evidence_estimated_input_tokens = getattr(token_count_resp_1, "total_tokens", None)
    except Exception:
        token_count_resp_1 = None

    try:
        evidence_response = await client.models.generate_content(
            model=model_id,
            contents=evidence_contents,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_json_schema=build_candidate_response_json_schema(requested_fields),
            ),
        )
    except Exception as exc:
        if is_retryable(exc):
            raise
        raise

    evidence_raw_text = evidence_response.text or "{}"
    evidence_candidates = coerce_candidate_response(
        json.loads(evidence_raw_text),
        requested_fields=requested_fields,
    )
    candidate_contexts = build_candidate_contexts(
        candidates=evidence_candidates,
        raw_lines=raw_lines,
    )

    resolution_prompt = build_resolution_prompt(
        requested_fields=requested_fields,
        candidates=evidence_candidates,
        candidate_contexts=candidate_contexts,
        few_shot_examples=few_shot_examples,
    )

    resolution_estimated_input_tokens = None
    try:
        token_count_resp_2 = await client.models.count_tokens(
            model=model_id,
            contents=resolution_prompt,
        )
        resolution_estimated_input_tokens = getattr(token_count_resp_2, "total_tokens", None)
    except Exception:
        token_count_resp_2 = None

    try:
        resolution_response = await client.models.generate_content(
            model=model_id,
            contents=resolution_prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_json_schema=build_final_response_json_schema(requested_fields),
            ),
        )
    except Exception as exc:
        if is_retryable(exc):
            raise
        raise

    resolution_raw_text = resolution_response.text or "{}"
    parsed_json = coerce_final_response(
        json.loads(resolution_raw_text),
        requested_fields=requested_fields,
    )
    extractions = make_extractions_from_json(
        parsed_json=parsed_json,
        requested_fields=requested_fields,
        source_documents=source_documents,
    )

    return SimpleNamespace(
        parsed_json=parsed_json,
        extractions=extractions,
        evidence_candidates=evidence_candidates,
        candidate_contexts=candidate_contexts,
        evidence_prompt=evidence_prompt,
        resolution_prompt=resolution_prompt,
        combined_ocr_text=combined_ocr_text,
        numbered_ocr_text=numbered_ocr_text,
        source_documents=source_documents,
        attached_pdf_files=pdf_files,
        estimated_input_tokens={
            "evidence": evidence_estimated_input_tokens,
            "resolution": resolution_estimated_input_tokens,
        },
        usage={
            "evidence": usage_to_dict(getattr(evidence_response, "usage_metadata", None)),
            "resolution": usage_to_dict(getattr(resolution_response, "usage_metadata", None)),
        },
        raw_response_texts={
            "evidence": evidence_raw_text,
            "resolution": resolution_raw_text,
        },
    )


async def extract_offer_documents_async(
    offer,
    api_key,
    requested_fields: dict[str, str] | None = None,
    few_shot_examples: list[Any] | None = None,
    concurrency=2,
    model_id=MODEL_ID,
):
    """
    Делает extraction в два шага:
    1. Ищет evidence-кандидатов по каждому полю.
    2. Резолвит кандидатов в финальный structured JSON.

    Поддерживает смешанный набор документов:
    - если у документа есть `pdf_url`, в модель отправляется сам PDF;
    - если `pdf_url` нет, в контекст добавляется его `ocr_text`.
    """
    del concurrency

    requested_fields = requested_fields or DEFAULT_FIELD_TYPES

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=120_000,
        ),
    ).aio

    try:
        combined_result = await extract_offer_async(
            client=client,
            offer=offer,
            requested_fields=requested_fields,
            few_shot_examples=few_shot_examples,
            model_id=model_id,
        )
    finally:
        await client.aclose()

    evidence_usage = combined_result.usage["evidence"]
    resolution_usage = combined_result.usage["resolution"]

    token_summary = {
        "estimated_input_tokens_sum": (
            (combined_result.estimated_input_tokens["evidence"] or 0)
            + (combined_result.estimated_input_tokens["resolution"] or 0)
        ),
        "prompt_token_count_sum": (
            (evidence_usage["prompt_token_count"] or 0)
            + (resolution_usage["prompt_token_count"] or 0)
        ),
        "candidates_token_count_sum": (
            (evidence_usage["candidates_token_count"] or 0)
            + (resolution_usage["candidates_token_count"] or 0)
        ),
        "total_token_count_sum": (
            (evidence_usage["total_token_count"] or 0)
            + (resolution_usage["total_token_count"] or 0)
        ),
        "thoughts_token_count_sum": (
            (evidence_usage["thoughts_token_count"] or 0)
            + (resolution_usage["thoughts_token_count"] or 0)
        ),
        "cached_content_token_count_sum": (
            (evidence_usage["cached_content_token_count"] or 0)
            + (resolution_usage["cached_content_token_count"] or 0)
        ),
    }

    return SimpleNamespace(
        parsed_json=combined_result.parsed_json,
        extractions=combined_result.extractions,
        evidence_candidates=combined_result.evidence_candidates,
        candidate_contexts=combined_result.candidate_contexts,
        per_document_results=[],
        combined_result=combined_result,
        errors=[],
        token_summary=token_summary,
        total_documents=len(offer.get("documents", [])),
        successful_documents=len(offer.get("documents", [])),
        failed_documents=0,
        requested_fields=requested_fields,
        few_shot_examples=few_shot_examples or [],
    )
