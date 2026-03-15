import asyncio
import io
import json
import logging
import mimetypes
import re
from types import SimpleNamespace
from typing import Any

import httpx
from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


MODEL_ID = "gemini-2.5-flash"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Дефолтный набор полей оставляем для быстрых локальных запусков.
DEFAULT_FIELD_TYPES = {
    "Spoluúčast": "string",
    "Smluvní pokuty": "string",
    "Územní rozsah": "string",
    "Roční pojistné": "string",
    "Regresní náhrady": "string",
    "Věci zaměstnanců": "string",
    "Asistenční služby": "string",
    "Dodatečné sublimity": "string",
    "Vyloučené činnosti": "string",
    "Osoby ve výkonu trestu": "string",
    "Krytí vadného výrobku": "string",
    "Limit nemajetkové újmy": "string",
    "Věci návštěv limit I": "string",
    "Čekací/karenční doba": "string",
    "Finanční škody limit I": "string",
    "Limit na věci převzaté": "string",
    "Osobnostní újma limit I": "string",
    "Vyloučené státy/sankce": "string",
    "Věci návštěv limit II": "string",
    "Finanční škody limit II": "string",
    "Limit pojistného plnění": "string",
    "Osobnostní újma limit II": "string",
    "Regres nemocenské limit I": "string",
    "Způsob stanovení prémia": "string",
    "Dvě a více spoluúčastí": "string",
    "Objasnění podpojištění": "string",
    "Odpovědnost za věci limit": "string",
    "Regres nemocenské limit II": "string",
    "Regres pojišťoven limit I": "string",
    "Věci zaměstnanců limit I": "string",
    "Obecná odpovědnost limit I": "string",
    "Regres pojišťoven limit II": "string",
    "Výkon trestu škody limit I": "string",
    "Věci zaměstnanců limit II": "string",
    "Obecná odpovědnost limit II": "string",
    "Výkon trestu škody limit II": "string",
    "Použití zvýšených limitů": "string",
    "Krytí subdodavatel/subdodávky": "string",
    "Výluky na kybernetická rizika": "string",
    "Věci návštěv spoluúčast I": "string",
    "Finanční škody spoluúčast I": "string",
    "Křížová odpovědnost limit I": "string",
    "Osobnostní újma spoluúčast I": "string",
    "Věci návštěv spoluúčast II": "string",
    "Finanční škody spoluúčast II": "string",
    "Křížová odpovědnost limit II": "string",
    "Osobnostní újma spoluúčast II": "string",
    "Regres nemocenské spoluúčast I": "string",
    "Limit čistých finančních škod": "string",
    "Regres nemocenské spoluúčast II": "string",
    "Regres pojišťoven spoluúčast I": "string",
    "Věci zaměstnanců spoluúčast I": "string",
    "Obecná odpovědnost spoluúčast I": "string",
    "Regres pojišťoven spoluúčast II": "string",
    "Výkon trestu škody spoluúčast I": "string",
    "Věci převzaté/užívané limit I": "string",
    "Věci zaměstnanců spoluúčast II": "string",
    "Obecná odpovědnost spoluúčast II": "string",
    "Výkon trestu škody spoluúčast II": "string",
    "Věci převzaté/užívané limit II": "string",
    "Křížová odpovědnost spoluúčast I": "string",
    "Křížová odpovědnost spoluúčast II": "string",
    "Odpovědnost za škodu vadnou činností": "string",
    "Věci převzaté/užívané spoluúčast I": "string",
    "Věci převzaté/užívané spoluúčast II": "string",
    "Limit na práce v rámci holdingové skupiny": "string",
}

DEFAULT_NUMERICISH_FIELDS = {
    "Roční pojistné",
    "Osoby ve výkonu trestu",
    "Limit pojistného plnění",
    "Regres pojišťoven limit I",
    "Obecná odpovědnost limit I",
    "Výkon trestu škody limit I",
    "Věci zaměstnanců limit II",
    "Obecná odpovědnost limit II",
    "Použití zvýšených limitů",
    "Křížová odpovědnost limit II",
}

NUMBER_TYPE_ALIASES = {"number", "int", "integer", "float", "numeric", "decimal"}
NUMBER_TOKEN_RE = re.compile(r"\d[\d\s.,\xa0]*")


def _json_pretty(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _normalize_example(example: Any) -> dict[str, Any]:
    if isinstance(example, dict):
        example_text = example["text"]
        requested_fields = example["requested_fields"]
        example_return = example.get("example_return", example.get("exaple_return"))
    elif isinstance(example, (list, tuple)) and len(example) == 3:
        example_text, requested_fields, example_return = example
    else:
        raise ValueError(
            "Few-shot example must be either a dict with keys "
            "'text', 'requested_fields', 'example_return' or a 3-item tuple/list."
        )

    if example_return is None:
        raise ValueError("Few-shot example is missing 'example_return'.")

    return {
        "text": example_text,
        "requested_fields": requested_fields,
        "example_return": example_return,
    }


def infer_numericish_fields(requested_fields: dict[str, str]) -> set[str]:
    numericish_fields = set()

    for field_name, field_type in requested_fields.items():
        normalized_type = str(field_type).strip().lower()
        if normalized_type in NUMBER_TYPE_ALIASES or field_name in DEFAULT_NUMERICISH_FIELDS:
            numericish_fields.add(field_name)

    return numericish_fields


def get_number_typed_fields(requested_fields: dict[str, str]) -> set[str]:
    return {
        field_name
        for field_name, field_type in requested_fields.items()
        if str(field_type).strip().lower() in NUMBER_TYPE_ALIASES
    }


def extract_first_number(value: Any) -> int | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value)

    match = NUMBER_TOKEN_RE.search(str(value))
    if not match:
        return None

    digits_only = re.sub(r"\D", "", match.group(0))
    if not digits_only:
        return None

    return int(digits_only)


def normalize_number_typed_fields(
    parsed_json: dict[str, Any],
    requested_fields: dict[str, str],
) -> dict[str, Any]:
    normalized = dict(parsed_json)

    for field_name in get_number_typed_fields(requested_fields):
        normalized[field_name] = extract_first_number(normalized.get(field_name))

    return normalized


def order_parsed_json(
    parsed_json: dict[str, Any],
    requested_fields: dict[str, str],
    none_placeholder: Any | None = None,
) -> dict[str, Any]:
    ordered: dict[str, Any] = {}

    for field_name in requested_fields:
        value = parsed_json.get(field_name)
        if value is None and none_placeholder is not None:
            value = none_placeholder
        ordered[field_name] = value

    return ordered


def build_output_offer_dict(
    offer: dict[str, Any],
    parsed_json: dict[str, Any],
    requested_fields: dict[str, str],
    none_placeholder: Any | None = None,
) -> dict[str, Any]:
    return {
        "id": offer.get("id"),
        "insurer": offer.get("insurer"),
        "label": offer.get("label"),
        **order_parsed_json(
            parsed_json=parsed_json,
            requested_fields=requested_fields,
            none_placeholder=none_placeholder,
        ),
    }


def build_response_json_schema(requested_fields: dict[str, str]) -> dict[str, Any]:
    properties = {}

    for field_name, loose_type in requested_fields.items():
        normalized_type = str(loose_type).strip().lower()

        if normalized_type in {"int", "integer"}:
            schema_type = ["integer", "number", "string", "null"]
        elif normalized_type in {"number", "float", "numeric", "decimal"}:
            schema_type = ["number", "string", "null"]
        elif normalized_type in {"bool", "boolean"}:
            schema_type = ["boolean", "string", "null"]
        elif normalized_type in {"array", "list"}:
            schema_type = ["array", "null"]
        elif normalized_type in {"object", "dict"}:
            schema_type = ["object", "null"]
        else:
            schema_type = ["string", "null"]

        properties[field_name] = {
            "type": schema_type,
            "description": (
                f"Extracted value for field '{field_name}'. "
                f"Requested loose type: {loose_type}."
            ),
        }

    return {
        "type": "object",
        "properties": properties,
        "required": list(requested_fields.keys()),
    }


def format_requested_fields(requested_fields: dict[str, str]) -> str:
    return "\n".join(
        f"- {field_name}: {field_type}"
        for field_name, field_type in requested_fields.items()
    )


def format_few_shot_examples(few_shot_examples: list[Any] | None) -> str:
    if not few_shot_examples:
        return "## FEW-SHOT EXAMPLES\nNo few-shot examples were provided."

    blocks = ["## FEW-SHOT EXAMPLES"]

    for index, raw_example in enumerate(few_shot_examples, start=1):
        example = _normalize_example(raw_example)
        blocks.append(f"### EXAMPLE {index}")
        blocks.append("EXAMPLE_TEXT:")
        blocks.append(example["text"])
        blocks.append("")
        blocks.append("EXAMPLE_REQUESTED_FIELDS:")
        blocks.append(_json_pretty(example["requested_fields"]))
        blocks.append("")
        blocks.append("EXAMPLE_OUTPUT_JSON:")
        blocks.append(_json_pretty(example["example_return"]))
        blocks.append("")

    return "\n".join(blocks).strip()


def format_document_manifest(source_documents: list[dict[str, Any]]) -> str:
    if not source_documents:
        return "[]"
    return _json_pretty(source_documents)


def build_prompt(
    requested_fields: dict[str, str],
    combined_ocr_text: str,
    source_documents: list[dict[str, Any]],
    few_shot_examples: list[Any] | None = None,
) -> str:
    numericish_fields = infer_numericish_fields(requested_fields)
    numericish_block = "\n".join(f"- {field_name}" for field_name in sorted(numericish_fields))

    if not numericish_block:
        numericish_block = "- none"

    return f"""
Z OCR textu v češtině vytěž strukturované informace o pojistné nabídce.

OCR může být špinavé, rozbité a bez původního formátování, proto:
- rekonstruuj význam podle okolního kontextu,
- správně mapuj limity, sublimity, spoluúčasti a pojistné ke správným polím,
- nevymýšlej údaje, které v textu nejsou,
- pokud hodnota není spolehlivě dohledatelná, vrať null,
- vracej pouze validní JSON podle zadaného schématu,
- pokud je textová hodnota příliš dlouhá, zkrať ji na stručné, ale věrné shrnutí.

Způsob vstupu dokumentů:
- Jako soubory se přikládají pouze skutečné PDF dokumenty.
- Pokud `pdf_url` ve skutečnosti ukazuje na jiný formát, například DOCX, soubor se nepřikládá a použije se pouze OCR text.
- OCR text je předán níže pro všechny dokumenty, i pro ty, které mají zároveň přiložené PDF.
- Použij všechny přiložené PDF dokumenty i všechny OCR texty společně jako jeden zdroj pravdy pro extrakci.
- Pokud je u stejného dokumentu k dispozici PDF i OCR text, ber je jako dva pohledy na tentýž dokument a využij oba společně.
- Pokud si PDF a OCR text odporují, preferuj informaci, která je v dokumentu explicitnější a spolehlivější.

Pravidla:
1. Zachovej přesně stejné názvy klíčů jako v requested_fields.
2. U číselných polí vrať pokud možno jen samotné číslo bez měny, jednotek a doprovodného textu; nikdy nevracej čistě slovní popis bez číslic.
3. U textových polí nevracej zbytečně dlouhé věty, pokud stačí kratší přesné shrnutí.
4. Pokud pole v dokumentech vůbec není nebo je nejednoznačné, vrať null.
5. Nepřidávej žádné další klíče mimo requested_fields.
6. Pokud název pole obsahuje "II", znamená to druhé relevantní uvedení nebo druhou variantu dané hodnoty v dokumentu; pole bez "II" obvykle odpovídá prvnímu uvedení.

Pole, která mají být číselná nebo číslo-obsahující:
{numericish_block}

{format_few_shot_examples(few_shot_examples)}

## CURRENT TASK
CURRENT_REQUESTED_FIELDS:
{_json_pretty(requested_fields)}

CURRENT_DOCUMENT_MANIFEST:
{format_document_manifest(source_documents)}

CURRENT_OCR_TEXT_FOR_ALL_DOCUMENTS:
{combined_ocr_text}
""".strip()


async def upload_pdf_from_url(client, pdf_url: str, filename: str) -> Any:
    async with httpx.AsyncClient(follow_redirects=True, timeout=120.0) as http_client:
        response = await http_client.get(pdf_url)
        response.raise_for_status()

    upload_mime_type = detect_upload_mime_type(
        filename=filename,
        content_type=response.headers.get("content-type"),
    )
    if upload_mime_type != "application/pdf":
        logger.info(
            "Skipping file attachment for filename=%s url=%s because it is not a real PDF (content_type=%s)",
            filename,
            pdf_url,
            response.headers.get("content-type"),
        )
        return None

    pdf_buffer = io.BytesIO(response.content)
    pdf_buffer.name = filename
    return await client.files.upload(
        file=pdf_buffer,
        config={"mime_type": upload_mime_type},
    )


async def build_offer_inputs(
    client,
    offer: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], list[Any]]:
    documents = offer.get("documents", [])
    combined_parts = []
    source_documents = []
    pdf_files = []

    for index, doc in enumerate(documents, start=1):
        filename = doc.get("filename") or f"document_{index}"
        document_id = doc.get("id")
        ocr_text = doc.get("ocr_text", "")
        pdf_url = doc.get("pdf_url")
        source_document = {
            "document_index": index - 1,
            "filename": filename,
            "document_id": document_id,
            "pdf_url": pdf_url,
        }

        if pdf_url:
            uploaded_pdf = await upload_pdf_from_url(
                client=client,
                pdf_url=pdf_url,
                filename=filename,
            )
            if uploaded_pdf is not None:
                pdf_files.append(uploaded_pdf)
                source_document["attached_as_pdf"] = True
            else:
                source_document["attached_as_pdf"] = False
        else:
            source_document["attached_as_pdf"] = False

        if source_document["attached_as_pdf"] and ocr_text:
            source_document["source_mode"] = "pdf+ocr_text"
        elif source_document["attached_as_pdf"]:
            source_document["source_mode"] = "pdf"
        else:
            source_document["source_mode"] = "ocr_text"

        if ocr_text:
            header_lines = [f"=== DOCUMENT {index} ===", f"filename: {filename}"]
            if document_id is not None:
                header_lines.append(f"document_id: {document_id}")
            header_lines.append(f"attached_as_pdf: {source_document['attached_as_pdf']}")
            header_lines.append("ocr_text:")
            combined_parts.append("\n".join(header_lines) + f"\n{ocr_text}")

        source_document["ocr_text_included"] = bool(ocr_text)

        source_documents.append(source_document)

    return "\n\n".join(combined_parts), source_documents, pdf_files


def coerce_parsed_json(parsed_json: Any, requested_fields: dict[str, str]) -> dict[str, Any]:
    if not isinstance(parsed_json, dict):
        raise ValueError("Model response is not a JSON object.")

    return {
        field_name: parsed_json.get(field_name)
        for field_name in requested_fields
    }


def make_extractions_from_json(
    parsed_json: dict[str, Any],
    requested_fields: dict[str, str] | None = None,
    source_documents: list[dict[str, Any]] | None = None,
):
    extractions = []

    for field_name, value in parsed_json.items():
        if value is None:
            continue

        extractions.append(
            SimpleNamespace(
                extraction_class="insurance_field",
                extraction_text=str(value),
                attributes={
                    "field_name": field_name,
                    "value_type": None if requested_fields is None else requested_fields.get(field_name),
                    "source_scope": "combined_offer_context",
                    "source_documents": source_documents or [],
                },
            )
        )

    return extractions


def usage_to_dict(usage_metadata):
    if not usage_metadata:
        return {
            "prompt_token_count": None,
            "candidates_token_count": None,
            "total_token_count": None,
            "thoughts_token_count": None,
            "cached_content_token_count": None,
        }

    return {
        "prompt_token_count": getattr(usage_metadata, "prompt_token_count", None),
        "candidates_token_count": getattr(usage_metadata, "candidates_token_count", None),
        "total_token_count": getattr(usage_metadata, "total_token_count", None),
        "thoughts_token_count": getattr(usage_metadata, "thoughts_token_count", None),
        "cached_content_token_count": getattr(usage_metadata, "cached_content_token_count", None),
    }


def is_retryable(exc: Exception) -> bool:
    error_text = str(exc)
    retry_markers = ["408", "429", "500", "502", "503", "504", "UNAVAILABLE", "DEADLINE_EXCEEDED"]
    return any(marker in error_text for marker in retry_markers)


def detect_upload_mime_type(filename: str | None, content_type: str | None) -> str | None:
    normalized_content_type = (content_type or "").split(";")[0].strip().lower()
    normalized_filename = (filename or "").strip().lower()
    guessed_mime_type, _ = mimetypes.guess_type(normalized_filename)

    if normalized_content_type == "application/pdf":
        return "application/pdf"
    if guessed_mime_type == "application/pdf":
        return "application/pdf"
    if normalized_filename.endswith(".pdf"):
        return "application/pdf"

    return None


def build_sort_params_response_json_schema(requested_fields: dict[str, str]) -> dict[str, Any]:
    numeric_fields = sorted(get_number_typed_fields(requested_fields))

    return {
        "type": "object",
        "properties": {
            "sort_params": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "column": {
                            "type": "string",
                            "enum": numeric_fields,
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["ASC", "DESC"],
                        },
                        "reason": {
                            "type": ["string", "null"],
                        },
                    },
                    "required": ["column", "direction", "reason"],
                },
            },
        },
        "required": ["sort_params"],
    }


def build_sort_params_prompt(requested_fields: dict[str, str]) -> str:
    numeric_fields = sorted(get_number_typed_fields(requested_fields))
    numeric_fields_block = "\n".join(f"- {field_name}" for field_name in numeric_fields)

    if not numeric_fields_block:
        numeric_fields_block = "- none"

    return f"""
You are choosing reranking sort parameters for insurance offers.

Task:
Return sort_params that can be used to sort offers from better to worse.

Rules:
1. Use only fields that are explicitly number-typed.
2. Higher is usually better for limits, sublimits, aggregate limits and multipliers.
3. Lower is usually better for premium, deductible, waiting period and penalty-like amounts.
4. If the direction is unclear, skip the field.
5. Order sort_params from more important to less important.
6. Return only valid JSON.

REQUESTED_FIELDS:
{_json_pretty(requested_fields)}

NUMBER_TYPED_FIELDS:
{numeric_fields_block}
""".strip()


def coerce_sort_params(parsed_json: Any, requested_fields: dict[str, str]) -> list[dict[str, Any]]:
    if not isinstance(parsed_json, dict):
        raise ValueError("Sort params response is not a JSON object.")

    numeric_fields = get_number_typed_fields(requested_fields)
    raw_sort_params = parsed_json.get("sort_params", [])
    normalized_sort_params = []

    for item in raw_sort_params:
        if not isinstance(item, dict):
            continue

        column = item.get("column")
        direction = str(item.get("direction", "DESC")).upper()
        if column not in numeric_fields:
            continue
        if direction not in {"ASC", "DESC"}:
            continue

        normalized_sort_params.append(
            {
                "column": column,
                "direction": direction,
                "reason": item.get("reason"),
            }
        )

    return normalized_sort_params


@retry(
    reraise=True,
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception(is_retryable),
)
async def extract_offer_async(
    client,
    offer: dict[str, Any],
    requested_fields: dict[str, str],
    few_shot_examples: list[Any] | None = None,
    model_id: str = MODEL_ID,
):
    offer_id = offer.get("id")
    logger.info("Starting extraction for offer_id=%s", offer_id)

    combined_ocr_text, source_documents, pdf_files = await build_offer_inputs(
        client=client,
        offer=offer,
    )
    response_json_schema = build_response_json_schema(requested_fields)
    prompt = build_prompt(
        requested_fields=requested_fields,
        combined_ocr_text=combined_ocr_text,
        source_documents=source_documents,
        few_shot_examples=few_shot_examples,
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
                "Falling back to OCR-only extraction for offer_id=%s because attached file processing failed: %s",
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
    parsed_json = order_parsed_json(parsed_json, requested_fields)
    extractions = make_extractions_from_json(
        parsed_json=parsed_json,
        requested_fields=requested_fields,
        source_documents=source_documents,
    )

    logger.info(
        "Extraction finished for offer_id=%s with %s non-null fields",
        offer_id,
        sum(1 for value in parsed_json.values() if value is not None),
    )

    return SimpleNamespace(
        parsed_json=parsed_json,
        extractions=extractions,
        prompt=prompt,
        combined_ocr_text=combined_ocr_text,
        source_documents=source_documents,
        attached_pdf_files=pdf_files,
        estimated_input_tokens=estimated_input_tokens,
        usage=usage_to_dict(getattr(response, "usage_metadata", None)),
        raw_response_text=raw_text,
    )


@retry(
    reraise=True,
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception(is_retryable),
)
async def extract_sort_params_async(
    client,
    requested_fields: dict[str, str],
    model_id: str = MODEL_ID,
):
    if not get_number_typed_fields(requested_fields):
        return SimpleNamespace(
            sort_params=[],
            prompt=build_sort_params_prompt(requested_fields),
            estimated_input_tokens=0,
            usage=usage_to_dict(None),
            raw_response_text='{"sort_params": []}',
        )

    prompt = build_sort_params_prompt(requested_fields)
    response_json_schema = build_sort_params_response_json_schema(requested_fields)

    estimated_input_tokens = None
    try:
        token_count_resp = await client.models.count_tokens(
            model=model_id,
            contents=prompt,
        )
        estimated_input_tokens = getattr(token_count_resp, "total_tokens", None)
    except Exception:
        token_count_resp = None

    try:
        response = await client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_json_schema=response_json_schema,
            ),
        )
    except Exception as exc:
        if is_retryable(exc):
            raise
        raise

    raw_text = response.text or '{"sort_params": []}'
    sort_params = coerce_sort_params(json.loads(raw_text), requested_fields)

    return SimpleNamespace(
        sort_params=sort_params,
        prompt=prompt,
        estimated_input_tokens=estimated_input_tokens,
        usage=usage_to_dict(getattr(response, "usage_metadata", None)),
        raw_response_text=raw_text,
    )


async def get_sort_params_async(
    api_key,
    requested_fields: dict[str, str] | None = None,
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
        return await extract_sort_params_async(
            client=client,
            requested_fields=requested_fields,
            model_id=model_id,
        )
    finally:
        await client.aclose()


def build_ranking_result(
    offers_parsed: list[dict[str, Any]],
    sort_params: list[dict[str, Any]],
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
    )
    best_offer_id = ranking[0] if ranking else None
    return ranking, best_offer_id


async def parse_and_rerank(
    input_data: dict[str, Any],
    api_key: str,
    few_shot_examples: list[Any] | None = None,
    concurrency: int = 2,
    model_id: str = MODEL_ID,
    include_debug_payload: bool = True,
) -> dict[str, Any]:
    offers = input_data.get("offers", [])
    field_types = input_data.get("field_types") or DEFAULT_FIELD_TYPES
    fields_to_extract = input_data.get("fields_to_extract")
    segment = input_data.get("segment")

    requested_fields = (
        {
            field_name: field_types[field_name]
            for field_name in fields_to_extract
        }
        if fields_to_extract
        else dict(field_types)
    )

    extraction_results = await extract_offers_documents_async(
        offers=offers,
        api_key=api_key,
        requested_fields=requested_fields,
        few_shot_examples=few_shot_examples,
        concurrency=concurrency,
        model_id=model_id,
    )

    ranking_ready_offers = [
        build_output_offer_dict(
            offer={
                "id": result.offer_id,
                "insurer": result.insurer,
                "label": result.label,
            },
            parsed_json=result.parsed_json,
            requested_fields=requested_fields,
        )
        for result in extraction_results
    ]

    sort_info = await get_sort_params_async(
        api_key=api_key,
        requested_fields=requested_fields,
        model_id=model_id,
    )

    ranking, best_offer_id = build_ranking_result(
        offers_parsed=ranking_ready_offers,
        sort_params=sort_info.sort_params,
    )

    offers_parsed = [
        build_output_offer_dict(
            offer={
                "id": result.offer_id,
                "insurer": result.insurer,
                "label": result.label,
            },
            parsed_json=result.parsed_json,
            requested_fields=requested_fields,
            none_placeholder="N/A",
        )
        for result in extraction_results
    ]

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
                "parsed_json": item.parsed_json,
                "errors": item.errors,
                "token_summary": item.token_summary,
            }
            for item in extraction_results
        ]

    return result


def _build_single_offer_result(
    offer: dict[str, Any],
    combined_result,
    requested_fields: dict[str, str],
    few_shot_examples: list[Any] | None,
):
    token_summary = {
        "estimated_input_tokens_sum": combined_result.estimated_input_tokens or 0,
        "prompt_token_count_sum": combined_result.usage["prompt_token_count"] or 0,
        "candidates_token_count_sum": combined_result.usage["candidates_token_count"] or 0,
        "total_token_count_sum": combined_result.usage["total_token_count"] or 0,
        "thoughts_token_count_sum": combined_result.usage["thoughts_token_count"] or 0,
        "cached_content_token_count_sum": combined_result.usage["cached_content_token_count"] or 0,
    }

    return SimpleNamespace(
        offer_id=offer.get("id"),
        insurer=offer.get("insurer"),
        label=offer.get("label"),
        parsed_json=order_parsed_json(combined_result.parsed_json, requested_fields),
        parsed_offer=build_output_offer_dict(
            offer=offer,
            parsed_json=combined_result.parsed_json,
            requested_fields=requested_fields,
        ),
        extractions=combined_result.extractions,
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


def _build_offer_error_result(
    offer: dict[str, Any],
    error: Exception,
    requested_fields: dict[str, str],
    few_shot_examples: list[Any] | None,
):
    return SimpleNamespace(
        offer_id=offer.get("id"),
        insurer=offer.get("insurer"),
        label=offer.get("label"),
        parsed_json=order_parsed_json(
            {field_name: None for field_name in requested_fields},
            requested_fields=requested_fields,
        ),
        parsed_offer=build_output_offer_dict(
            offer=offer,
            parsed_json={field_name: None for field_name in requested_fields},
            requested_fields=requested_fields,
        ),
        extractions=[],
        per_document_results=[],
        combined_result=None,
        errors=[
            {
                "offer_id": offer.get("id"),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        ],
        token_summary={
            "estimated_input_tokens_sum": 0,
            "prompt_token_count_sum": 0,
            "candidates_token_count_sum": 0,
            "total_token_count_sum": 0,
            "thoughts_token_count_sum": 0,
            "cached_content_token_count_sum": 0,
        },
        total_documents=len(offer.get("documents", [])),
        successful_documents=0,
        failed_documents=len(offer.get("documents", [])),
        requested_fields=requested_fields,
        few_shot_examples=few_shot_examples or [],
    )


async def extract_offers_documents_async(
    offers,
    api_key,
    requested_fields: dict[str, str] | None = None,
    few_shot_examples: list[Any] | None = None,
    concurrency=2,
    model_id=MODEL_ID,
):
    requested_fields = requested_fields or DEFAULT_FIELD_TYPES
    semaphore = asyncio.Semaphore(concurrency)

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=120_000,
        ),
    ).aio

    async def run_one(single_offer):
        async with semaphore:
            return await extract_offer_async(
                client=client,
                offer=single_offer,
                requested_fields=requested_fields,
                few_shot_examples=few_shot_examples,
                model_id=model_id,
            )

    try:
        raw_results = await asyncio.gather(
            *(run_one(single_offer) for single_offer in offers),
            return_exceptions=True,
        )
    finally:
        await client.aclose()

    results = []
    for single_offer, item in zip(offers, raw_results):
        if isinstance(item, Exception):
            logger.error(
                "Offer extraction failed for offer_id=%s insurer=%s label=%s error_type=%s error=%s",
                single_offer.get("id"),
                single_offer.get("insurer"),
                single_offer.get("label"),
                type(item).__name__,
                str(item),
                exc_info=(type(item), item, item.__traceback__),
            )
            results.append(
                _build_offer_error_result(
                    offer=single_offer,
                    error=item,
                    requested_fields=requested_fields,
                    few_shot_examples=few_shot_examples,
                )
            )
            continue

        logger.info(
            "Offer extraction completed for offer_id=%s insurer=%s label=%s",
            single_offer.get("id"),
            single_offer.get("insurer"),
            single_offer.get("label"),
        )
        results.append(
            _build_single_offer_result(
                offer=single_offer,
                combined_result=item,
                requested_fields=requested_fields,
                few_shot_examples=few_shot_examples,
            )
        )

    return results


async def extract_offer_documents_async(
    offer,
    api_key,
    requested_fields: dict[str, str] | None = None,
    few_shot_examples: list[Any] | None = None,
    concurrency=2,
    model_id=MODEL_ID,
):
    """
    Объединяет все OCR документы оффера в один контекст и делает один structured-output вызов.

    few_shot_examples ожидаются в формате:
    [
        {
            "text": "...",
            "requested_fields": {"Pole A": "string", "Pole B": "number"},
            "example_return": {"Pole A": "...", "Pole B": 123}
        }
    ]

    Дополнительно поддерживается ключ "exaple_return" для совместимости с ранними черновиками.
    """
    if isinstance(offer, (list, tuple)):
        return await extract_offers_documents_async(
            offers=offer,
            api_key=api_key,
            requested_fields=requested_fields,
            few_shot_examples=few_shot_examples,
            concurrency=concurrency,
            model_id=model_id,
        )

    requested_fields = requested_fields or DEFAULT_FIELD_TYPES

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=120_000,
        ),
    ).aio

    try:
        try:
            combined_result = await extract_offer_async(
                client=client,
                offer=offer,
                requested_fields=requested_fields,
                few_shot_examples=few_shot_examples,
                model_id=model_id,
            )
        except Exception as exc:
            logger.error(
                "Single-offer extraction failed for offer_id=%s insurer=%s label=%s error_type=%s error=%s",
                offer.get("id"),
                offer.get("insurer"),
                offer.get("label"),
                type(exc).__name__,
                str(exc),
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            raise
    finally:
        await client.aclose()

    logger.info(
        "Single-offer extraction completed for offer_id=%s insurer=%s label=%s",
        offer.get("id"),
        offer.get("insurer"),
        offer.get("label"),
    )
    return _build_single_offer_result(
        offer=offer,
        combined_result=combined_result,
        requested_fields=requested_fields,
        few_shot_examples=few_shot_examples,
    )


RESPONSE_JSON_SCHEMA = build_response_json_schema(DEFAULT_FIELD_TYPES)
PROMPT = build_prompt(
    requested_fields=DEFAULT_FIELD_TYPES,
    combined_ocr_text="",
    source_documents=[],
    few_shot_examples=None,
)
