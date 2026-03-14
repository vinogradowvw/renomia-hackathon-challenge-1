from google.genai import types
from pydantic import BaseModel
from textwrap import dedent
from typing import Optional

from agents.config import GeminiTracker


class LimitsParsed(BaseModel):
    """Structured extraction result for limits fields."""

    basic_limit_czk: Optional[int] = None
    limit_multiplier_per_year: Optional[int] = None
    aggregate_limit_czk: Optional[int] = None
    limit_persons_in_custody_czk: Optional[int] = None
    limit_pure_financial_loss_czk: Optional[int] = None
    limit_taken_items_czk: Optional[int] = None
    limit_cross_liability_czk: Optional[int] = None
    limit_recourse_czk: Optional[int] = None
    limit_non_pecuniary_damage_czk: Optional[int] = None
    premium_czk: Optional[int] = None

LIMIT_PARSER_SYSTEM_PROMT = """
Jsi specializovaný extrakční model pro české dokumenty pojištění odpovědnosti.

Tvým úkolem je z poskytnutého OCR textu extrahovat pouze limity pojistného plnění do přesně jednoho JSON objektu podle zadaného schématu.

Extrahuj pouze tato pole:
- basic_limit_czk
- limit_multiplier_per_year
- aggregate_limit_czk
- limit_persons_in_custody_czk
- limit_pure_financial_loss_czk
- limit_taken_items_czk
- limit_cross_liability_czk
- limit_recourse_czk
- limit_non_pecuniary_damage_czk

Důležitá pravidla:
- Vrať pouze validní JSON objekt.
- Nepřidávej žádný text před JSON ani za JSON.
- Nepřidávej markdown.
- Nepřidávej žádné komentáře ani poznámky.
- Nepřidávej žádné klíče mimo zadané schéma.
- Pokud údaj není v textu výslovně uveden nebo jej nelze spolehlivě určit, vrať null.
- Nevymýšlej hodnoty.
- Nezaměňuj limity se spoluúčastí, pojistným ani jinými částkami.
- `premium_czk` je částka, kterou klient platí za pojištění, například „pojistné“, „roční pojistné“, „běžné pojistné“ nebo „celkové roční pojistné“.
- Proto nikdy nepoužívej částku pojistného jako limit plnění ani jako sublimit.
- Nezaměňuj obecný limit za sublimit, pokud text výslovně neříká, že se vztahuje k danému poli.
- Pokud dodatky nebo zvláštní ujednání mění původní limit, použij konečnou upravenou hodnotu, pokud je to z textu jasné.
- Upřednostňuj tabulky limitů, přehledy krytí, dodatky a zvláštní ujednání před obecným popisem produktu.

Mapování polí:
- basic_limit_czk = hlavní limit odpovědnostního pojištění, například „limit plnění“, „základní limit“, „pojistná částka“, pokud jde o hlavní limit odpovědnosti
- limit_multiplier_per_year = počet obnovení limitu za pojistný rok nebo roční násobek limitu
- aggregate_limit_czk = celkový roční agregovaný limit
- limit_persons_in_custody_czk = sublimit pro osoby v péči
- limit_pure_financial_loss_czk = sublimit pro čistou finanční újmu
- limit_taken_items_czk = sublimit pro převzaté, svěřené, vnesené, užívané nebo držené věci, pouze pokud je tato vazba v textu jasná
- limit_cross_liability_czk = sublimit pro křížovou odpovědnost
- limit_recourse_czk = sublimit pro regres nebo regresní nároky
- limit_non_pecuniary_damage_czk = sublimit pro nemajetkovou újmu

Za významově příbuzné nebo ekvivalentní výrazy považuj podle kontextu zejména:
- „limit plnění“, „limit pojistného plnění“, „pojistná částka“, „základní limit“
- „roční agregát“, „agregovaný limit“, „celkový roční limit“
- „pojistné“, „roční pojistné“, „běžné pojistné“, „celkové roční pojistné“ jako výrazy pro premium_czk, nikoli pro limity
- „čistá finanční újma“
- „nemajetková újma“
- „regres“, „regresní nároky“
- „křížová odpovědnost“
- „převzaté věci“, „věci převzaté“, „věci užívané“, „vnesené věci“
- „osoby v péči“

Normalizace čísel:
- Všechny peněžní částky vrať jako integer v Kč bez mezer, teček, čárek a měnových symbolů.
- Příklady:
  - „50 000 000 Kč“ -> 50000000
  - „50.000.000,- Kč“ -> 50000000
  - „10 tis. Kč“ -> 10000
  - „2,5 mil. Kč“ -> 2500000
- Pokud je roční agregovaný limit výslovně uveden, použij jej do aggregate_limit_czk.
- Pokud je uveden pouze basic_limit_czk a současně je jasně uveden limit_multiplier_per_year, můžeš odvodit aggregate_limit_czk jako basic_limit_czk * limit_multiplier_per_year.
- Žádné jiné číselné pole neodvozuj.
- Pokud není jasné, zda částka patří k hlavnímu limitu nebo jen k sublimitu, nehádej a vrať null pro nejasné pole.

Práce s nejednoznačností:
- Pokud se v textu objeví více kandidátních hodnot, vyber tu, která je nejpříměji navázaná na pojištění odpovědnosti v analyzované nabídce.
- Pokud je částka uvedena pouze obecně bez jasné vazby na některé z výše uvedených polí, nepřiřazuj ji.
- Pokud text obsahuje více verzí limitu a jedna z nich je zjevně výsledná nebo upravená dodatkem, použij výslednou hodnotu.

Výstup musí odpovídat přesně tomuto schématu:
{
  "basic_limit_czk": int | null,
  "limit_multiplier_per_year": int | null,
  "aggregate_limit_czk": int | null,
  "limit_persons_in_custody_czk": int | null,
  "limit_pure_financial_loss_czk": int | null,
  "limit_taken_items_czk": int | null,
  "limit_cross_liability_czk": int | null,
  "limit_recourse_czk": int | null,
  "limit_non_pecuniary_damage_czk": int | null
}
"""


class LimitParser:

    def __init__(self, gemini: GeminiTracker) -> None:
        self.gemini = gemini

    def parse_from_chunks(self, chunks: list[str]):
        all_chunks = "## ALL CHUNKS:"
        for i, c in enumerate(chunks):
            all_chunks += f"\n ### CHUNK No {i}: \n {c}"
        prompt = dedent(LIMIT_PARSER_SYSTEM_PROMT + all_chunks).strip()

        response = self.gemini.generate(
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                )
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=LimitsParsed,
                temperature=0,
            ),
        )
        return self.__model_from_response(response)

    def __model_from_response(self, response) -> LimitsParsed:
        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            if isinstance(parsed, LimitsParsed):
                return parsed
            return LimitsParsed.model_validate(parsed)
        return LimitsParsed.model_validate_json(response.text)
