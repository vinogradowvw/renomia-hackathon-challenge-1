from google import genai
from google.genai import types
import threading
import os
from typing import Optional
from textwrap import dedent
from config import config

from pydantic import BaseModel, ConfigDict, Field


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


class OfferDocumentInput(BaseModel):
    filename: str
    ocr_text: str


class OfferInput(BaseModel):
    """Single offer payload sent to the extraction model."""

    id: str
    insurer: str
    label: str
    documents: list[OfferDocumentInput]


class ExtractionInput(BaseModel):
    """Model input for parsing one insurance offer within a segment."""
    offer: OfferInput
    segment: str


class ParsedOutput(BaseModel):
    """Structured extraction result for one insurance offer."""

    covered_activities: Optional[str] = None
    territorial_scope: Optional[str] = None
    basic_limit_czk: Optional[int] = None
    limit_multiplier_per_year: Optional[int] = None
    aggregate_limit_czk: Optional[int] = None
    limit_persons_in_custody_czk: Optional[int] = None
    limit_pure_financial_loss_czk: Optional[int] = None
    limit_taken_items_czk: Optional[int] = None
    limit_cross_liability_czk: Optional[int] = None
    limit_recourse_czk: Optional[int] = None
    limit_non_pecuniary_damage_czk: Optional[int] = None
    basic_deductible_czk: Optional[int] = None
    deductible_recourse_czk: Optional[int] = None
    deductible_non_pecuniary_czk: Optional[int] = None
    deductible_brought_items_czk: Optional[int] = None
    deductible_financial_loss_czk: Optional[int] = None
    premium_czk: Optional[int] = Field(default=None)

class OfferParsed(BaseModel):
    """Structured extraction result for one insurance offer."""

    model_config = ConfigDict(extra="forbid")

    id: str
    insurer: str
    label: str
    covered_activities: Optional[str] = None
    territorial_scope: Optional[str] = None
    basic_limit_czk: Optional[int] = None
    limit_multiplier_per_year: Optional[int] = None
    aggregate_limit_czk: Optional[int] = None
    limit_persons_in_custody_czk: Optional[int] = None
    limit_pure_financial_loss_czk: Optional[int] = None
    limit_taken_items_czk: Optional[int] = None
    limit_cross_liability_czk: Optional[int] = None
    limit_recourse_czk: Optional[int] = None
    limit_non_pecuniary_damage_czk: Optional[int] = None
    basic_deductible_czk: Optional[int] = None
    deductible_recourse_czk: Optional[int] = None
    deductible_non_pecuniary_czk: Optional[int] = None
    deductible_brought_items_czk: Optional[int] = None
    deductible_financial_loss_czk: Optional[int] = None
    premium_czk: Optional[int] = Field(default=None)




generate_content_config = genai.types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=ParsedOutput,
)
# =============================================================================



# =============================================================================
EXTRACTION_SYSTEM_PROMPT = """
You are a strict extraction engine for liability insurance OCR text.

Return exactly one valid JSON object matching the provided schema.
Return JSON only. Do not output explanations, notes, markdown, comments, or extra keys.

This request contains one OCR chunk from one liability insurance offer.
Extract only information explicitly supported by this chunk.
Do not use prior knowledge, assumptions from other chunks, or typical insurance defaults.
If a value is not explicitly present in this chunk, return null.

Accuracy is more important than recall.
However, when a value is clearly and explicitly stated in the chunk, extract it confidently.
Do not ignore clear values just because wording is imperfect or OCR is noisy.

Tolerate common OCR artifacts such as:
- missing Czech diacritics
- broken spacing
- punctuation noise
- `Kc` instead of `Kč`
- `spoluucast` instead of `spoluúčast`
- number formatting variants with spaces, dots, or commas

Use conservative matching.
If OCR corruption makes a value ambiguous, return null.

Treat the following Czech insurance terms and close OCR variants as equivalent when the meaning is clear:
- `limit plneni`, `pojistna castka`, `zakladni limit`, `limit pojistneho plneni`
- `spoluucast`
- `rocni pojistne`, `bezne pojistne`, `celkove rocni pojistne`
- `uzemni rozsah`
- `cista financni ujma`
- `nemajetkova ujma`
- `regres`
- `krizova odpovednost`
- `prevzate veci`, `veci prevzate`, `veci uzivane`, `vnesene veci`
- `osoby v peci`

Return all money fields as integer CZK amounts without separators or currency symbols.
Normalize only values explicitly stated in the chunk.
Do not calculate, estimate, or derive missing values.
Examples:
- `50 000 000 Kc` -> 50000000
- `50.000.000,- Kc` -> 50000000
- `10 tis. Kc` -> 10000
- `2,5 mil. Kc` -> 2500000

If a multiplier per insurance year is explicitly stated, extract it into `limit_multiplier_per_year`.
If an annual aggregate limit is explicitly stated, extract it into `aggregate_limit_czk`.
Do not convert installment premium into annual premium unless the annual premium is explicitly stated.

Field mapping rules:
- `covered_activities`: concise plain-language summary of insured activities and explicit notable exclusions only if clearly stated in this chunk.
- `territorial_scope`: exact geographic scope supported by the chunk, as a comma-separated string.
  Rules:
  - If specific countries are explicitly listed, return those countries only.
  - If a whole region is explicitly stated, return that region exactly.
  - Do not expand a region into countries.
  - Do not compress listed countries into a broader region.
  - Do not generalize beyond the text.
- `basic_limit_czk`: main liability limit explicitly applicable to the insured liability coverage.
- `limit_multiplier_per_year`: explicit annual reinstatement count or annual limit multiple.
- `aggregate_limit_czk`: explicit total annual aggregate limit.
- `limit_persons_in_custody_czk`: explicit sublimit for persons in care/custody.
- `limit_pure_financial_loss_czk`: explicit sublimit for pure financial loss.
- `limit_taken_items_czk`: explicit sublimit for taken, entrusted, brought, used, or held items only if clearly mapped.
- `limit_cross_liability_czk`: explicit sublimit for cross liability.
- `limit_recourse_czk`: explicit sublimit for recourse/regress.
- `limit_non_pecuniary_damage_czk`: explicit sublimit for non-pecuniary damage.
- `basic_deductible_czk`: standard deductible.
- `deductible_recourse_czk`: deductible for recourse/regress.
- `deductible_non_pecuniary_czk`: deductible for non-pecuniary damage.
- `deductible_brought_items_czk`: deductible for brought / entrusted / taken items where clearly applicable.
- `deductible_financial_loss_czk`: deductible for financial loss.
- `premium_czk`: explicitly stated annual premium only.

Do not confuse:
- limits with deductibles
- per-event / per-claim limits with annual aggregate limits
- general policy limits with sublimits

If several candidate values appear and cannot be chosen unambiguously from this chunk, return null.
If an endorsement or modification clearly overrides a base value inside this chunk, use the final modified value.
""".strip()




# =============================================================================
class GeminiTracker:
    """Wrapper around Gemini that tracks token usage."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        config: types.GenerateContentConfig | None = None,
    ):
        self.enabled = bool(api_key)
        self.model_name = model_name
        self.config = config

        if self.enabled:
            self.client = genai.Client(api_key=api_key)

        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.request_count = 0
        self._lock = threading.Lock()

    def generate(self, prompt=None, **kwargs):
        if not self.enabled:
            raise RuntimeError("Gemini API key not configured")

        response = self.client.models.generate_content(
            model=self.model_name,
            config=self.config,
            **kwargs,
        )

        with self._lock:
            self.request_count += 1
            meta = getattr(response, "usage_metadata", None)
            if meta:
                self.prompt_tokens += getattr(meta, "prompt_token_count", 0) or 0
                self.completion_tokens += getattr(meta, "candidates_token_count", 0) or 0
                self.total_tokens += getattr(meta, "total_token_count", 0) or 0
        return response

    def get_metrics(self):
        with self._lock:
            return {
                "gemini_request_count": self.request_count,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            }

    def reset(self):
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            self.request_count = 0

gemini = GeminiTracker(GEMINI_API_KEY, config=generate_content_config)
