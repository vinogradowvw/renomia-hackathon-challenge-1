from google.genai import types
from pydantic import BaseModel
from textwrap import dedent
from typing import Optional

from agents.config import GeminiTracker


class DeductionsParsed(BaseModel):
    """Structured extraction result for deductible fields."""

    basic_deductible_czk: Optional[int] = None
    deductible_recourse_czk: Optional[int] = None
    deductible_non_pecuniary_czk: Optional[int] = None
    deductible_brought_items_czk: Optional[int] = None
    deductible_financial_loss_czk: Optional[int] = None


DEDUCTION_PARSER_SYSTEM_PROMT = """
Jsi specializovaný extrakční model pro české dokumenty pojištění odpovědnosti.

Tvým úkolem je z poskytnutého OCR textu extrahovat pouze spoluúčasti do přesně jednoho JSON objektu podle zadaného schématu.

Extrahuj pouze tato pole:
- basic_deductible_czk
- deductible_recourse_czk
- deductible_non_pecuniary_czk
- deductible_brought_items_czk
- deductible_financial_loss_czk

Důležitá pravidla:
- Vrať pouze validní JSON objekt.
- Nepřidávej žádný text před JSON ani za JSON.
- Nepřidávej markdown.
- Nepřidávej žádné komentáře ani poznámky.
- Nepřidávej žádné klíče mimo zadané schéma.
- Pokud údaj není v textu výslovně uveden nebo jej nelze spolehlivě určit, vrať null.
- Nevymýšlej hodnoty.
- Nezaměňuj spoluúčast s limitem, pojistným ani jinými částkami.
- Nezaměňuj obecnou spoluúčast za speciální spoluúčast, pokud text výslovně neříká, že se vztahuje k danému poli.
- Pokud dodatky nebo zvláštní ujednání mění původní spoluúčast, použij konečnou upravenou hodnotu, pokud je to z textu jasné.
- Upřednostňuj tabulky spoluúčastí, přehledy krytí, dodatky a zvláštní ujednání před obecným popisem produktu.

Mapování polí:
- basic_deductible_czk = základní spoluúčast, obecná spoluúčast pro hlavní odpovědnostní krytí
- deductible_recourse_czk = spoluúčast pro regres nebo regresní nároky
- deductible_non_pecuniary_czk = spoluúčast pro nemajetkovou újmu
- deductible_brought_items_czk = spoluúčast pro převzaté, svěřené, vnesené, užívané nebo držené věci, pouze pokud je tato vazba v textu jasná
- deductible_financial_loss_czk = spoluúčast pro čistou finanční újmu nebo finanční škodu, pokud je tato vazba v textu jasná

Za významově příbuzné nebo ekvivalentní výrazy považuj podle kontextu zejména:
- „spoluúčast“, „sjednaná spoluúčast“, „spoluúčast pojištěného“
- „regres“, „regresní nároky“
- „nemajetková újma“
- „čistá finanční újma“, „finanční újma“, „finanční škoda“
- „převzaté věci“, „věci převzaté“, „věci užívané“, „vnesené věci“

Normalizace čísel:
- Všechny peněžní částky vrať jako integer v Kč bez mezer, teček, čárek a měnových symbolů.
- Příklady:
  - „10 000 Kč“ -> 10000
  - „10.000,- Kč“ -> 10000
  - „5 tis. Kč“ -> 5000
  - „2,5 tis. Kč“ -> 2500
- Neodvozuj žádná čísla, pokud nejsou v textu výslovně uvedena.
- Pokud není jasné, zda částka patří k obecné nebo speciální spoluúčasti, nehádej a vrať null pro nejasné pole.

Práce s nejednoznačností:
- Pokud se v textu objeví více kandidátních hodnot, vyber tu, která je nejpříměji navázaná na pojištění odpovědnosti v analyzované nabídce.
- Pokud je částka uvedena pouze obecně bez jasné vazby na některé z výše uvedených polí, nepřiřazuj ji.
- Pokud text obsahuje více verzí spoluúčasti a jedna z nich je zjevně výsledná nebo upravená dodatkem, použij výslednou hodnotu.

Výstup musí odpovídat přesně tomuto schématu:
{
  "basic_deductible_czk": int | null,
  "deductible_recourse_czk": int | null,
  "deductible_non_pecuniary_czk": int | null,
  "deductible_brought_items_czk": int | null,
  "deductible_financial_loss_czk": int | null
}
"""


class DeductionParser:

    def __init__(self, gemini: GeminiTracker) -> None:
        self.gemini = gemini

    def parse_from_chunks(self, chunks: list[str]):
        all_chunks = "## ALL CHUNKS:"
        for i, c in enumerate(chunks):
            all_chunks += f"\n ### CHUNK No {i}: \n {c}"
        prompt = dedent(DEDUCTION_PARSER_SYSTEM_PROMT + all_chunks).strip()

        response = self.gemini.generate(
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                )
            ]
        )
        return self.__model_from_response(response)

    def __model_from_response(self, response) -> DeductionsParsed:
        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            if isinstance(parsed, DeductionsParsed):
                return parsed
            return DeductionsParsed.model_validate(parsed)

        return DeductionsParsed.model_validate_json(response.text)
