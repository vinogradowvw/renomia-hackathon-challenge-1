from types import SimpleNamespace
from typing import Any

import json
from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from extraction_prototype import (
    DEFAULT_FIELD_TYPES,
    MODEL_ID,
    _json_pretty,
    build_offer_inputs,
    build_response_json_schema,
    coerce_parsed_json,
    format_document_manifest,
    format_few_shot_examples,
    infer_numericish_fields,
    is_retryable,
    logger,
    normalize_number_typed_fields,
    usage_to_dict,
)


def _normalize_langextract_values(langextract_values: Any) -> dict[str, list[str]]:
    if isinstance(langextract_values, dict):
        normalized: dict[str, list[str]] = {}
        for field_name, values in langextract_values.items():
            if isinstance(values, list):
                normalized[field_name] = [str(value) for value in values if value not in (None, "")]
            elif values not in (None, ""):
                normalized[field_name] = [str(values)]
        return normalized

    extracted_values = getattr(langextract_values, "extracted_values", None)
    if isinstance(extracted_values, dict):
        return _normalize_langextract_values(extracted_values)

    return {}


def build_judge_prompt(
    requested_fields: dict[str, str],
    combined_ocr_text: str,
    source_documents: list[dict[str, Any]],
    langextract_values: dict[str, list[str]],
    few_shot_examples: list[Any] | None = None,
) -> str:
    numericish_fields = infer_numericish_fields(requested_fields)
    numericish_block = "\n".join(f"- {field_name}" for field_name in sorted(numericish_fields))

    if not numericish_block:
        numericish_block = "- none"

    return f"""
You are the final judge for insurance field extraction.

Your job is to verify and correct the preliminary LangExtract output against the actual source documents.
The LangExtract output is only a candidate signal, not ground truth.

Input handling:
- Only real PDF documents may be attached as files.
- OCR text is provided for all documents that have it.
- Use both attached PDF documents and OCR text together as the evidence base.

Task:
1. Review the LangExtract candidates field by field.
2. Check whether each candidate is actually supported by the source documents.
3. Correct wrong candidates when the source documents clearly support a better value.
4. If LangExtract missed a field but the source documents clearly contain it, fill it in.
5. Return the final answer strictly in the requested JSON format.

Rules:
1. Keep exactly the same keys as in requested_fields.
2. Do not invent values that are not supported by the documents.
3. Treat LangExtract output as suggestions only; you may keep, replace or discard them.
4. For number-typed fields, return only the numeric value when possible.
5. For text fields, if the value would be too long, shorten it into a concise but faithful summary.
6. If a field remains unclear after checking the documents, return null.
7. If the field name contains "II", it usually refers to the second relevant occurrence or second variant of the value.
8. If multiple LangExtract candidates exist for one field, choose the best document-supported value or create a numeric range.
9. Return valid JSON only.

Requested fields:
{_json_pretty(requested_fields)}

Numeric-like fields:
{numericish_block}

{format_few_shot_examples(few_shot_examples)}

LANGEXTRACT_CANDIDATES:
{_json_pretty(langextract_values)}

CURRENT_DOCUMENT_MANIFEST:
{format_document_manifest(source_documents)}

CURRENT_OCR_TEXT_FOR_ALL_DOCUMENTS:
{combined_ocr_text}
""".strip()


@retry(
    reraise=True,
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception(is_retryable),
)
async def judge_offer_async(
    client,
    offer: dict[str, Any],
    langextract_values: Any,
    requested_fields: dict[str, str],
    few_shot_examples: list[Any] | None = None,
    model_id: str = MODEL_ID,
):
    offer_id = offer.get("id")
    logger.info("Starting judge step for offer_id=%s", offer_id)

    combined_ocr_text, source_documents, pdf_files = await build_offer_inputs(
        client=client,
        offer=offer,
    )
    normalized_langextract_values = _normalize_langextract_values(langextract_values)
    response_json_schema = build_response_json_schema(requested_fields)
    prompt = build_judge_prompt(
        requested_fields=requested_fields,
        combined_ocr_text=combined_ocr_text,
        source_documents=source_documents,
        langextract_values=normalized_langextract_values,
        few_shot_examples=few_shot_examples,
    )
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
                "Falling back to OCR-only judge step for offer_id=%s because attached file processing failed: %s",
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
        "Judge step finished for offer_id=%s with %s non-null fields",
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
        langextract_values=normalized_langextract_values,
        estimated_input_tokens=estimated_input_tokens,
        usage=usage_to_dict(getattr(response, "usage_metadata", None)),
        raw_response_text=raw_text,
    )


async def judge_offer_documents_async(
    offer: dict[str, Any],
    langextract_values: Any,
    api_key: str,
    requested_fields: dict[str, str] | None = None,
    few_shot_examples: list[Any] | None = None,
    model_id: str = MODEL_ID,
):
    requested_fields = requested_fields or DEFAULT_FIELD_TYPES

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=120_000,
        ),
    ).aio

    try:
        return await judge_offer_async(
            client=client,
            offer=offer,
            langextract_values=langextract_values,
            requested_fields=requested_fields,
            few_shot_examples=few_shot_examples,
            model_id=model_id,
        )
    finally:
        await client.aclose()
