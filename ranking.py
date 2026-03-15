from __future__ import annotations

import re
from typing import Any

import pandas as pd


class Ranking:
    _NULL_TOKENS = {"", "none", "null", "nan", "neuvedeno", "n/a", "-"}

    def _normalize_rules(self, sort_params: list[Any]) -> list[tuple[str, bool]]:
        rules: list[tuple[str, bool]] = []
        for item in sort_params:
            if isinstance(item, dict):
                column = item.get("column")
                direction = str(item.get("direction", "DESC")).upper()
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                column = str(item[0])
                direction = str(item[1]).upper()
            else:
                continue

            if not column:
                continue
            is_asc = direction == "ASC"
            rules.append((column, is_asc))
        return rules

    def _parse_numeric_value(self, value: Any) -> float | None:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)

        text = str(value).strip().lower().replace("\u00a0", " ")
        if text in self._NULL_TOKENS:
            return None

        multiplier = 1.0
        if re.search(r"\bmil\.?\b|\bmilion", text):
            multiplier = 1_000_000.0
        elif re.search(r"\btis\.?\b|\btisic", text):
            multiplier = 1_000.0

        # Keep digits, separators and sign; remove currency and words.
        s = re.sub(r"[^0-9,.\-]", "", text)
        # Drop separators that are not between digits (OCR noise like "tis.").
        s = re.sub(r"(?<!\d)[.,]|[.,](?!\d)", "", s)
        # Keep minus only if it is leading.
        if "-" in s[1:]:
            s = s[0] + s[1:].replace("-", "") if s.startswith("-") else s.replace("-", "")
        if not s or s in {"-", ".", ","}:
            return None

        # If both separators are present, treat the last one as decimal separator.
        if "," in s and "." in s:
            last_comma = s.rfind(",")
            last_dot = s.rfind(".")
            decimal_sep = "," if last_comma > last_dot else "."
            thousands_sep = "." if decimal_sep == "," else ","
            s = s.replace(thousands_sep, "")
            if decimal_sep == ",":
                s = s.replace(",", ".")
        elif "," in s:
            if s.count(",") > 1:
                s = s.replace(",", "")
            else:
                left, right = s.split(",", 1)
                s = left + right if len(right) == 3 else left + "." + right
        elif "." in s:
            if s.count(".") > 1:
                s = s.replace(".", "")
            else:
                left, right = s.split(".", 1)
                s = left + right if len(right) == 3 else left + "." + right

        try:
            return float(s) * multiplier
        except ValueError:
            return None

    def _parse_numeric_series(self, series: pd.Series) -> pd.Series:
        parsed = series.map(self._parse_numeric_value)
        return pd.to_numeric(parsed, errors="coerce")

    def rank(self, df: pd.DataFrame, sort_params: list[Any], null_threshold: float = 0.67) -> list[str]:
        """
        Rank rows by count of best values across numeric columns.

        For each numeric rule:
        - DESC => best value is column maximum
        - ASC  => best value is column minimum

        Final ordering:
        1) best_values_count DESC
        2) tie-break by provided numeric sort rules in order

        sort_params supports either:
        - [{"column": "basic_limit_czk", "direction": "DESC"}, ...]
        - [("basic_limit_czk", "DESC"), ...]

        null_threshold:
        - if null share in a column is greater than this value, the column is ignored.
        """
        if df.empty:
            return []

        rules = self._normalize_rules(sort_params)
        if not rules:
            return df["id"].astype(str).tolist() if "id" in df.columns else df.index.astype(str).tolist()

        ranked_df = df.copy()
        sort_columns: list[str] = []
        ascending_flags: list[bool] = []

        for column, is_asc in rules:
            if column not in ranked_df.columns:
                continue

            numeric_series = self._parse_numeric_series(ranked_df[column])
            if not numeric_series.notna().any():
                continue
            if float(numeric_series.isna().mean()) > null_threshold:
                continue

            ranked_df[column] = numeric_series
            sort_columns.append(column)
            ascending_flags.append(is_asc)

        if not sort_columns:
            return ranked_df["id"].astype(str).tolist() if "id" in ranked_df.columns else ranked_df.index.astype(str).tolist()

        best_count = pd.Series(0, index=ranked_df.index, dtype="int64")
        for column, is_asc in zip(sort_columns, ascending_flags):
            series = ranked_df[column]
            non_null = series.dropna()
            if non_null.empty:
                continue
            best_value = non_null.min() if is_asc else non_null.max()
            best_count += (series.sub(best_value).abs() <= 1e-9).fillna(False).astype("int64")

        ranked_df["__best_values_count"] = best_count
        ranked_df = ranked_df.sort_values(
            by=["__best_values_count", *sort_columns],
            ascending=[False, *ascending_flags],
            na_position="last",
            kind="mergesort",
        ).drop(columns=["__best_values_count"])
        return ranked_df["id"].astype(str).tolist() if "id" in ranked_df.columns else ranked_df.index.astype(str).tolist()
