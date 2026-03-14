from google.genai import types
from pydantic import BaseModel
from textwrap import dedent

from agents.config import GeminiTracker


class CoverageParsed(BaseModel):
    """Structured extraction result for coverage summary fields."""

    covered_activities: str
    territorial_scope: str


COVERAGE_PARSER_SYSTEM_PROMT = """
Jsi specializovaný extrakční model pro české dokumenty pojištění odpovědnosti.

Tvým úkolem je z poskytnutého OCR textu extrahovat pouze dvě textová pole do přesně jednoho JSON objektu podle zadaného schématu.

Extrahuj pouze tato pole:
- covered_activities
- territorial_scope

Důležitá pravidla:
- Vrať pouze validní JSON objekt.
- Nepřidávej žádný text před JSON ani za JSON.
- Nepřidávej markdown.
- Nepřidávej žádné komentáře ani poznámky.
- Nepřidávej žádné klíče mimo zadané schéma.
- Nevymýšlej informace, které nejsou v textu podložené.
- Pokud je pole nejasné nebo v textu chybí, vrať co nejstručnější opatrný text založený jen na tom, co je v dokumentu zřejmé.
- Upřednostňuj text pojistné smlouvy, tabulek krytí, zvláštních ujednání, dodatků a výluk před marketingovým popisem produktu.

Mapování polí:
- covered_activities = stručný, věcný souhrn pojištěných činností a důležitých výluk nebo omezení, pokud jsou v textu jasně uvedeny
- territorial_scope = stručný souhrn územní platnosti pojištění

Pravidla pro covered_activities:
- Shrň pojištěné činnosti do krátkého srozumitelného textu.
- Pokud dokument uvádí výluky nebo omezení, zahrň pouze ty podstatné a jasně formulované.
- Nepřepisuj dlouhé pasáže doslova, spíš je zhušti do stručného souhrnu.
- Pokud text obsahuje jen obecné označení produktu bez konkrétního popisu činností, použij opatrný stručný popis založený na dostupném textu.

Pravidla pro territorial_scope:
- Vrať stručný textový výstup, ideálně seznam území oddělený čárkami, pokud to text umožňuje.
- Normalizuj běžné tvary do stručné podoby, například:
  - „Česká republika“ -> „ČR“
  - „Slovenská republika“ -> „SR“
  - „Evropa“ ponech jako „Evropa“
  - „celý svět kromě USA a Kanady“ ponech jako stručný věcný text
- Pokud dokument obsahuje územní omezení nebo výluky, zahrň je stručně do výsledku.

Práce s nejednoznačností:
- Pokud se v textu objeví více variant, vyber tu, která je nejpříměji navázaná na konkrétní analyzovanou nabídku.
- Pokud dodatky nebo zvláštní ujednání mění rozsah činností nebo územní rozsah, použij výslednou upravenou verzi, pokud je to z textu jasné.
- Pokud nelze spolehlivě odlišit pojištěné činnosti od obecného marketingového popisu, vrať co nejopatrnější stručné shrnutí.

Výstup musí odpovídat přesně tomuto schématu:
{
  "covered_activities": "string",
  "territorial_scope": "string"
}
"""


class CoverageParser:

    def __init__(self, gemini: GeminiTracker) -> None:
        self.gemini = gemini

    def parse_from_chunks(self, chunks: list[str]):
        all_chunks = "## ALL CHUNKS:"
        for i, c in enumerate(chunks):
            all_chunks += f"\n ### CHUNK No {i}: \n {c}"
        prompt = dedent(COVERAGE_PARSER_SYSTEM_PROMT + all_chunks).strip()

        response = self.gemini.generate(
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                )
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CoverageParsed,
                temperature=0,
            ),
        )
        return self.__model_from_response(response)

    def __model_from_response(self, response) -> CoverageParsed:
        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            if isinstance(parsed, CoverageParsed):
                return parsed
            return CoverageParsed.model_validate(parsed)

        return CoverageParsed.model_validate_json(response.text)
