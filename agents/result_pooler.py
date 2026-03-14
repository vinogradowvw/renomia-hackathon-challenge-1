from typing import Any

from pydantic import BaseModel

from agents.config import ParsedOutput


class ResultPooler:
    """Runs routed chunks through individual parsers and merges their outputs."""

    def __init__(self, parsers: list[object]) -> None:
        self.parsers = {
            self._parser_name(parser): parser
            for parser in parsers
        }

    def pool(self, route_result: dict[str, list[str]]) -> ParsedOutput:
        pooled_result: dict[str, Any] = {
            "covered_activities": "",
            "territorial_scope": "",
            "basic_limit_czk": None,
            "limit_multiplier_per_year": None,
            "aggregate_limit_czk": None,
            "limit_persons_in_custody_czk": None,
            "limit_pure_financial_loss_czk": None,
            "limit_taken_items_czk": None,
            "limit_cross_liability_czk": None,
            "limit_recourse_czk": None,
            "limit_non_pecuniary_damage_czk": None,
            "basic_deductible_czk": None,
            "deductible_recourse_czk": None,
            "deductible_non_pecuniary_czk": None,
            "deductible_brought_items_czk": None,
            "deductible_financial_loss_czk": None,
            "premium_czk": None,
        }

        for parser_name, chunks in route_result.items():
            if not chunks:
                continue

            parser = self.parsers.get(parser_name)
            if parser is None:
                continue

            parsed_result = parser.parse_from_chunks(chunks)
            parsed_payload = self._result_to_dict(parsed_result)
            pooled_result.update(parsed_payload)

        return ParsedOutput.model_validate(pooled_result)

    @staticmethod
    def _parser_name(parser: object) -> str:
        return getattr(parser, "name", parser.__class__.__name__)

    @staticmethod
    def _result_to_dict(result: BaseModel | dict[str, Any]) -> dict[str, Any]:
        if isinstance(result, BaseModel):
            return result.model_dump()
        if isinstance(result, dict):
            return result
        raise TypeError(f"Unsupported parser result type: {type(result).__name__}")
