import json
from collections import Counter
from textwrap import dedent
from typing import Any
import logging

from google.genai import types
from pydantic import BaseModel

from agents.config import (
    EXTRACTION_SYSTEM_PROMPT,
    ExtractionInput,
    GeminiTracker,
    OfferParsed,
    ParsedOutput
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHUNK_SIZE_TOKENS = 4000
CHUNK_OVERLAP_TOKENS = 400


class OfferChunk(BaseModel):
    source_id: str
    chunk_id: str
    text: str


def _tokenize(text: str) -> list[str]:
    return text.split()


def build_offer_chunks(
    payload: ExtractionInput,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[OfferChunk]:
    chunks: list[OfferChunk] = []
    step = max(1, chunk_size - overlap)

    for doc_index, document in enumerate(payload.offer.documents):
        source_id = document.filename or f"document_{doc_index}"
        tokens = _tokenize(document.ocr_text or "")
        if not tokens:
            continue

        chunk_index = 0
        for start in range(0, len(tokens), step):
            end = min(len(tokens), start + chunk_size)
            chunk_tokens = tokens[start:end]
            if not chunk_tokens:
                continue

            chunks.append(
                OfferChunk(
                    source_id=source_id,
                    chunk_id=f"{source_id}:{chunk_index}",
                    text=" ".join(chunk_tokens),
                )
            )
            chunk_index += 1

            if end >= len(tokens):
                break

    return chunks


def build_extraction_user_prompt(payload: ExtractionInput, chunk: OfferChunk) -> str:
    prompt_payload = {
        "segment": payload.segment,
        "offer": {
            "id": payload.offer.id,
            "insurer": payload.offer.insurer,
            "label": payload.offer.label,
        },
        "chunk": {
            "source_id": chunk.source_id,
            "chunk_id": chunk.chunk_id,
        },
    }

    return dedent(
        f"""
        Extract fields from this single OCR chunk.

        OFFER METADATA:
        {json.dumps(prompt_payload, ensure_ascii=False, indent=2)}

        OCR CHUNK:
        {chunk.text}
        """
    ).strip()


def _parse_chunk(payload: ExtractionInput, chunk: OfferChunk, model: GeminiTracker) -> ParsedOutput:
    response = model.generate(
        contents=build_extraction_user_prompt(payload, chunk),
        config=types.GenerateContentConfig(
            system_instruction=EXTRACTION_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=ParsedOutput,
            temperature=0,
        ),
    )

    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, ParsedOutput):
        return parsed
    if parsed is not None:
        try:
            return ParsedOutput.model_validate(parsed)
        except Exception:
            pass

    return ParsedOutput.model_validate_json(response.text)


def _resolve_numeric(values: list[Any]) -> int | None:
    non_null = [value for value in values if value is not None]
    if not non_null:
        return None

    counts = Counter(non_null)
    top_count = max(counts.values())
    winners = [value for value, count in counts.items() if count == top_count]
    if len(winners) == 1:
        return winners[0]
    return None


def _resolve_string(values: list[Any]) -> str | None:
    non_null = [str(value).strip() for value in values if value not in (None, "")]
    if not non_null:
        return None

    counts = Counter(non_null)
    top_count = max(counts.values())
    winners = [value for value, count in counts.items() if count == top_count]
    if len(winners) == 1:
        return winners[0]
    return None


def merge_chunk_outputs(chunk_outputs: list[ParsedOutput]) -> ParsedOutput:
    if not chunk_outputs:
        return ParsedOutput()

    merged: dict[str, Any] = {}
    for field_name in ParsedOutput.model_fields:
        values = [getattr(item, field_name) for item in chunk_outputs]
        if field_name in {"covered_activities", "territorial_scope"}:
            merged[field_name] = _resolve_string(values)
        else:
            merged[field_name] = _resolve_numeric(values)

    return ParsedOutput(**merged)

def parse_offer(payload: ExtractionInput, model: GeminiTracker) -> OfferParsed:
    logger.info(f"Parsing offer {payload.offer.id} with {len(payload.offer.documents)} documents")
    
    chunks = build_offer_chunks(payload)
    chunk_outputs = [_parse_chunk(payload, chunk, model) for chunk in chunks]
    merged = merge_chunk_outputs(chunk_outputs)

    logger.info(f"Parsed offer {payload.offer.id}: {merged}")
    
    return OfferParsed(
        id=payload.offer.id,
        insurer=payload.offer.insurer,
        label=payload.offer.label,
        covered_activities=merged.covered_activities,
        territorial_scope=merged.territorial_scope,
        basic_limit_czk=merged.basic_limit_czk,
        limit_multiplier_per_year=merged.limit_multiplier_per_year,
        aggregate_limit_czk=merged.aggregate_limit_czk,
        limit_persons_in_custody_czk=merged.limit_persons_in_custody_czk,
        limit_pure_financial_loss_czk=merged.limit_pure_financial_loss_czk,
        limit_taken_items_czk=merged.limit_taken_items_czk,
        limit_cross_liability_czk=merged.limit_cross_liability_czk,
        limit_recourse_czk=merged.limit_recourse_czk,
        limit_non_pecuniary_damage_czk=merged.limit_non_pecuniary_damage_czk,
        basic_deductible_czk=merged.basic_deductible_czk,
        deductible_recourse_czk=merged.deductible_recourse_czk,
        deductible_non_pecuniary_czk=merged.deductible_non_pecuniary_czk,
        deductible_brought_items_czk=merged.deductible_brought_items_czk,
        deductible_financial_loss_czk=merged.deductible_financial_loss_czk,
        premium_czk=merged.premium_czk,
    )
