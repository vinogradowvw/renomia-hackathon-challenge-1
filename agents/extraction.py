import json
from textwrap import dedent
from google.genai import types

from agents.config import (
    EXTRACTION_SYSTEM_PROMPT,
    ExtractionInput,
    GeminiTracker,
    OfferParsed,
    generate_content_config,
)


def build_extraction_prompt(payload: ExtractionInput) -> str:
    documents_text = "\n\n".join(
        f"[Document: {document.filename}]\n{document.ocr_text}"
        for document in payload.offer.documents
    )

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

    ## DOCUMENT TEXT:
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
        return OfferParsed.model_validate(parsed)

    return OfferParsed.model_validate_json(response.text)
