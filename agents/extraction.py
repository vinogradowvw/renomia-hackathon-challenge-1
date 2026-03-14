import json
from textwrap import dedent
from google.genai import types

from config import EXTRACTION_SYSTEM_PROMPT, GeminiTracker, OfferDocumentInput, OfferInput

from agents.config import ExtractionInput, OfferParsedOutput, generate_content_config
def build_extraction_prompt(payload: OfferInput, document_idx) -> str:
    prompt_payload = {
        "id": payload.offer.id,
        "insurer": payload.offer.insurer,
        "label": payload.offer.label,
        "document_name": payload.documents[document_idx].filename
    }

    return dedent(EXTRACTION_SYSTEM_PROMPT + f"""
    ## OFFER METADATA:
    {json.dumps(prompt_payload, ensure_ascii=False, indent=2)}

    ## DOCUMENT TEXT: 
    {payload.documents[document_idx].ocr_text}
    """).strip()


def parse_offer(payload: ExtractionInput, ai_client: GeminiTracker) -> OfferParsedOutput:
    response = ai_client.generate(
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=build_extraction_prompt(payload))],
            )
        ],
        config=types.GenerateContentConfig(
            system_instruction=EXTRACTION_SYSTEM_PROMPT,
            response_mime_type=generate_content_config.response_mime_type,
            response_schema=generate_content_config.response_schema,
        ),
    )

    if getattr(response, "parsed", None):
        parsed = response.parsed
        if isinstance(parsed, OfferParsedOutput):
            return parsed
        return OfferParsedOutput.model_validate(parsed)

    return OfferParsedOutput.model_validate_json(response.text)