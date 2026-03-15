from __future__ import annotations

import json
import os
import time
import unittest
import unicodedata
from statistics import mean
from typing import Any

from extraction_prototype import parse_and_rerank
from gemini_extraction_judge import pipeline as judge_pipeline


def _normalize_segment_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    without_diacritics = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    return without_diacritics.casefold()


def _require_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise unittest.SkipTest(
            "Set GEMINI_API_KEY or GOOGLE_API_KEY to run the parser comparison test."
        )
    return api_key


def _load_training_data() -> list[dict[str, Any]]:
    try:
        import psycopg2
    except ImportError as exc:
        raise unittest.SkipTest("psycopg2 is required for this integration test.") from exc

    dsn = os.getenv("TRAINING_DB_DSN")
    if dsn:
        conn = psycopg2.connect(dsn)
    else:
        required_env = {
            "host": os.getenv("TRAINING_DB_HOST"),
            "port": os.getenv("TRAINING_DB_PORT"),
            "user": os.getenv("TRAINING_DB_USER"),
            "password": os.getenv("TRAINING_DB_PASSWORD"),
            "dbname": os.getenv("TRAINING_DB_NAME"),
        }
        missing = [name for name, value in required_env.items() if not value]
        if missing:
            raise unittest.SkipTest(
                "Set TRAINING_DB_DSN or TRAINING_DB_HOST/TRAINING_DB_PORT/"
                "TRAINING_DB_USER/TRAINING_DB_PASSWORD/TRAINING_DB_NAME to run this test."
            )
        conn = psycopg2.connect(**required_env)

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT input, expected_output FROM training_data WHERE challenge_id = 1"
            )
            all_rows = cur.fetchall()

    cases = [
        {
            "input": input_data,
            "expected_output": expected_output,
        }
        for input_data, expected_output in all_rows
    ]

    requested_segment = os.getenv("TRAINING_SEGMENT")
    if requested_segment:
        normalized_requested_segment = _normalize_segment_key(requested_segment)
        cases = [
            case
            for case in cases
            if _normalize_segment_key(case["input"].get("segment", "")) == normalized_requested_segment
        ]

    case_limit = int(os.getenv("TRAINING_CASE_LIMIT", "1"))
    if case_limit > 0:
        cases = cases[:case_limit]

    if not cases:
        raise unittest.SkipTest("No training cases matched the current filters.")

    return cases


def _normalize_value_for_distance(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _normalized_levenshtein_distance(left: str, right: str) -> float:
    if left == right:
        return 0.0
    if not left and not right:
        return 0.0
    if not left or not right:
        return 1.0

    previous_row = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current_row = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            insert_cost = current_row[right_index - 1] + 1
            delete_cost = previous_row[right_index] + 1
            replace_cost = previous_row[right_index - 1] + (left_char != right_char)
            current_row.append(min(insert_cost, delete_cost, replace_cost))
        previous_row = current_row

    edit_distance = previous_row[-1]
    return edit_distance / max(len(left), len(right), 1)


def _compute_average_field_distance(
    actual_output: dict[str, Any],
    expected_output: dict[str, Any],
    field_names: list[str],
) -> float:
    actual_by_id = {
        str(offer.get("id")): offer
        for offer in actual_output.get("offers_parsed", [])
    }
    expected_by_id = {
        str(offer.get("id")): offer
        for offer in expected_output.get("offers_parsed", [])
    }

    distances: list[float] = []

    for offer_id, expected_offer in expected_by_id.items():
        actual_offer = actual_by_id.get(offer_id, {})
        for field_name in field_names:
            expected_value = _normalize_value_for_distance(expected_offer.get(field_name))
            actual_value = _normalize_value_for_distance(actual_offer.get(field_name))
            distances.append(
                _normalized_levenshtein_distance(actual_value, expected_value)
            )

    return mean(distances) if distances else 0.0


class CompareParsersIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_compare_gemini_extraction_judge_vs_extraction_prototype(self):
        api_key = _require_api_key()
        cases = _load_training_data()

        report: dict[str, Any] = {
            "case_count": len(cases),
            "cases": [],
            "extraction_prototype": {
                "avg_field_distance": None,
                "total_parse_seconds": 0.0,
                "ranking_exact_matches": 0,
                "best_offer_exact_matches": 0,
            },
            "gemini_extraction_judge": {
                "avg_field_distance": None,
                "total_parse_seconds": 0.0,
                "ranking_exact_matches": 0,
                "best_offer_exact_matches": 0,
            },
        }

        prototype_distances: list[float] = []
        judge_distances: list[float] = []

        for index, case in enumerate(cases, start=1):
            input_data = case["input"]
            expected_output = case["expected_output"]
            field_names = input_data.get("fields_to_extract") or list(input_data["field_types"].keys())

            prototype_started_at = time.perf_counter()
            prototype_output = await parse_and_rerank(
                input_data=input_data,
                api_key=api_key,
                include_debug_payload=False,
            )
            prototype_elapsed = time.perf_counter() - prototype_started_at

            judge_started_at = time.perf_counter()
            judge_output = await judge_pipeline(
                input_data=input_data,
                api_key=api_key,
                include_debug_payload=False,
                use_cache=False,
            )
            judge_elapsed = time.perf_counter() - judge_started_at

            self.assertEqual(len(prototype_output["offers_parsed"]), len(input_data["offers"]))
            self.assertEqual(len(judge_output["offers_parsed"]), len(input_data["offers"]))

            prototype_distance = _compute_average_field_distance(
                actual_output=prototype_output,
                expected_output=expected_output,
                field_names=field_names,
            )
            judge_distance = _compute_average_field_distance(
                actual_output=judge_output,
                expected_output=expected_output,
                field_names=field_names,
            )

            prototype_distances.append(prototype_distance)
            judge_distances.append(judge_distance)

            report["extraction_prototype"]["total_parse_seconds"] += prototype_elapsed
            report["gemini_extraction_judge"]["total_parse_seconds"] += judge_elapsed

            if prototype_output.get("ranking") == expected_output.get("ranking"):
                report["extraction_prototype"]["ranking_exact_matches"] += 1
            if judge_output.get("ranking") == expected_output.get("ranking"):
                report["gemini_extraction_judge"]["ranking_exact_matches"] += 1

            if prototype_output.get("best_offer_id") == expected_output.get("best_offer_id"):
                report["extraction_prototype"]["best_offer_exact_matches"] += 1
            if judge_output.get("best_offer_id") == expected_output.get("best_offer_id"):
                report["gemini_extraction_judge"]["best_offer_exact_matches"] += 1

            report["cases"].append(
                {
                    "index": index,
                    "segment": input_data.get("segment"),
                    "offers": len(input_data.get("offers", [])),
                    "fields": len(field_names),
                    "extraction_prototype": {
                        "avg_field_distance": prototype_distance,
                        "parse_seconds": prototype_elapsed,
                        "ranking": prototype_output.get("ranking"),
                        "best_offer_id": prototype_output.get("best_offer_id"),
                    },
                    "gemini_extraction_judge": {
                        "avg_field_distance": judge_distance,
                        "parse_seconds": judge_elapsed,
                        "ranking": judge_output.get("ranking"),
                        "best_offer_id": judge_output.get("best_offer_id"),
                    },
                }
            )

        report["extraction_prototype"]["avg_field_distance"] = (
            mean(prototype_distances) if prototype_distances else 0.0
        )
        report["gemini_extraction_judge"]["avg_field_distance"] = (
            mean(judge_distances) if judge_distances else 0.0
        )

        print(json.dumps(report, ensure_ascii=False, indent=2))

        self.assertGreaterEqual(
            report["extraction_prototype"]["avg_field_distance"],
            0.0,
        )
        self.assertGreaterEqual(
            report["gemini_extraction_judge"]["avg_field_distance"],
            0.0,
        )
