from google import genai
import threading
import os
from typing import Optional
from textwrap import dedent

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

    model_config = ConfigDict(extra="forbid")

    covered_activities: str
    territorial_scope: str
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
    covered_activities: str
    territorial_scope: str
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
EXTRACTION_SYSTEM_PROMPT = dedent(
    """
    You are an expert insurance document extraction engine liability insurance.
    Your task is to read raw OCR text from one insurance offer and
    return exactly one structured JSON object matching the provided schema.

    Your primary goal is accurate field extraction, not summarization. Extract only what is supported
    by the source text. Do not invent values. If a field is missing, unclear, contradictory, or cannot
    be mapped with high confidence, return null for numeric fields and a short cautious summary for
    string fields only when the text supports it.

    Follow these rules carefully:

    1. Scope of extraction
    - Extract data for one offer only.
    - Use all provided OCR text from all documents belonging to that offer.
    - Prefer explicit policy schedule tables, limits tables, endorsements, and premium sections over
      marketing language or generic product descriptions.

    2. insurance vocabulary
    - Treat related Czech wording as equivalent when appropriate:
      `limit plneni`, `pojistna castka`, `zakladni limit`, `limit pojistneho plneni`
      `spoluucast`
      `rocni pojistne`, `bezne pojistne`, `celkove rocni pojistne`
      `uzemni rozsah`
      `cista financni ujma`
      `nemajetkova ujma`
      `regres`
      `krizova odpovednost`
      `prevzate veci`, `veci prevzate`, `veci uzivane`, `vnesene veci`
      `osoby v peci`

    3. Numeric normalization
    - Return all money fields as integer CZK amounts without separators or currency symbols.
    - Examples:
      `50 000 000 Kc` -> 50000000
      `50.000.000,- Kc` -> 50000000
      `10 tis. Kc` -> 10000
      `2,5 mil. Kc` -> 2500000
    - If a limit is written as a multiple per insurance year, extract the multiplier into
      `limit_multiplier_per_year`.
    - If the annual aggregate limit is explicitly stated, use it for `aggregate_limit_czk`.
    - If only a base limit and annual multiplier are given, infer aggregate_limit_czk as
      `basic_limit_czk * limit_multiplier_per_year`.
    - Do not infer any other numeric field unless the document clearly states the equivalence.

    5. Field mapping
    - `covered_activities`: concise plain-language summary of insured activities and notable exclusions
      only if clearly stated in the OCR text.
    - `territorial_scope`: concise summary of the geographic validity exactly as supported by the text (fill as a string - list of items, comma separated).
    - `basic_limit_czk`: main policy liability limit.
    - `limit_multiplier_per_year`: annual reinstatement count or annual limit multiple.
    - `aggregate_limit_czk`: total annual aggregate limit.
    - `limit_persons_in_custody_czk`: sublimit for persons in care/custody.
    - `limit_pure_financial_loss_czk`: sublimit for pure financial loss.
    - `limit_taken_items_czk`: sublimit for taken, entrusted, brought, used, or held items only if
      the text clearly maps to this field.
    - `limit_cross_liability_czk`: sublimit for cross liability.
    - `limit_recourse_czk`: sublimit for recourse/regress.
    - `limit_non_pecuniary_damage_czk`: sublimit for non-pecuniary damage.
    - `basic_deductible_czk`: standard deductible.
    - `deductible_recourse_czk`: deductible for recourse/regress.
    - `deductible_non_pecuniary_czk`: deductible for non-pecuniary damage.
    - `deductible_brought_items_czk`: deductible for brought/entrusted/taken items where applicable.
    - `deductible_financial_loss_czk`: deductible for financial loss.
    - `premium_czk`: annual premium. Prefer annualized premium if multiple payment periods appear.

    6. Ambiguity handling
    - Do not confuse limits with deductibles.
    - Do not confuse per-event limits with annual aggregates.
    - Do not copy a general policy limit into a sublimit field unless the text explicitly applies it.
    - If several candidate values appear, choose the value most directly tied to liability coverage for
      the insured offer being analyzed.
    - If endorsements modify a base policy, prefer the final modified value when clearly stated.

    7. Output requirements
    - Match the schema exactly.
    - Do not include explanations, notes, markdown, or extra keys.
    """
).strip()


# =============================================================================
class GeminiTracker:
    """Wrapper around Gemini that tracks token usage."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        self.enabled = bool(api_key)
        if self.enabled:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.request_count = 0
        self._lock = threading.Lock()

    def generate(self, prompt = None, **kwargs):
        if not self.enabled:
            raise RuntimeError("Gemini API key not configured")
        if prompt is None:
            response = self.model.generate_content(**kwargs)
        else:
            response = self.model.generate_content(prompt, **kwargs)
        with self._lock:
            self.request_count += 1
            meta = getattr(response, "usage_metadata", None)
            if meta:
                self.prompt_tokens += getattr(meta, "prompt_token_count", 0)
                self.completion_tokens += getattr(meta, "candidates_token_count", 0)
                self.total_tokens += getattr(meta, "total_token_count", 0)
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


gemini = GeminiTracker(GEMINI_API_KEY)
