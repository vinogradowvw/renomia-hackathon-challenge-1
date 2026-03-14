import json
from textwrap import dedent
from google.genai import types

from agents.config import (
    EXTRACTION_SYSTEM_PROMPT,
    ExtractionInput,
    GeminiTracker,
    OfferParsed,
    ParsedOutput
)


def build_extraction_prompt(payload: ExtractionInput) -> str:
    documents_text = "\n\n".join(
        f"[Document: {document.filename}]\n{document.ocr_text}"
        for document in payload.offer.documents
    )

    print("")
    print("")
    print("-"*80)
    print(documents_text.count("[Document: "))

    prompt_payload = {
        "segment": payload.segment,
        "offer": {
            "id": payload.offer.id,
            "insurer": payload.offer.insurer,
            "label": payload.offer.label,
        },
    }

    return dedent(
        EXTRACTION_SYSTEM_PROMPT
        + f"""

    ## OFFER METADATA:
    {json.dumps(prompt_payload, ensure_ascii=False, indent=2)}

    ## DOCUMENT TEXTS:
    {documents_text}
    """
    ).strip()


def parse_offer(payload: ExtractionInput, model: GeminiTracker) -> OfferParsed:
    response = model.generate(
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=build_extraction_prompt(payload))],
            )
        ]
    )

    if getattr(response, "parsed", None):
        parsed = response.parsed
        if isinstance(parsed, OfferParsed):
            return parsed
        offer_parsed = ParsedOutput.model_validate_json(response.text)

        result = OfferParsed(
            id=payload.offer.id,
            insurer=payload.offer.insurer,
            label=payload.offer.label,
            covered_activities=offer_parsed.covered_activities,
            territorial_scope=offer_parsed.territorial_scope,
            basic_limit_czk=offer_parsed.basic_limit_czk,
            limit_multiplier_per_year=offer_parsed.limit_multiplier_per_year,
            aggregate_limit_czk=offer_parsed.aggregate_limit_czk,
            limit_persons_in_custody_czk=offer_parsed.limit_persons_in_custody_czk,
            limit_pure_financial_loss_czk=offer_parsed.limit_pure_financial_loss_czk,
            limit_taken_items_czk=offer_parsed.limit_taken_items_czk,
            limit_cross_liability_czk=offer_parsed.limit_cross_liability_czk,
            limit_recourse_czk=offer_parsed.limit_recourse_czk,
            limit_non_pecuniary_damage_czk=offer_parsed.limit_non_pecuniary_damage_czk,
            basic_deductible_czk=offer_parsed.basic_deductible_czk,
            deductible_recourse_czk=offer_parsed.deductible_recourse_czk,
            deductible_non_pecuniary_czk=offer_parsed.deductible_non_pecuniary_czk,
            deductible_brought_items_czk=offer_parsed.deductible_brought_items_czk,
            deductible_financial_loss_czk=offer_parsed.deductible_financial_loss_czk,
            premium_czk=offer_parsed.premium_czk,
        )
        return result