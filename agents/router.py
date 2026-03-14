import math
import re
import unicodedata
from typing import Any

import numpy as np

try:
    from simplemma import lemmatize as simplemma_lemmatize
except ModuleNotFoundError:
    simplemma_lemmatize = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer
except ModuleNotFoundError:
    SklearnTfidfVectorizer = None

from agents.config import gemini
from agents.parsers.coverage_parser import COVERAGE_PARSER_SYSTEM_PROMT
from agents.parsers.coverage_parser import CoverageParser
from agents.parsers.deduction_parser import DEDUCTION_PARSER_SYSTEM_PROMT
from agents.parsers.deduction_parser import DeductionParser
from agents.parsers.limit_parser import LIMIT_PARSER_SYSTEM_PROMT
from agents.parsers.limit_parser import LimitParser


TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ÿ]+", re.UNICODE)


class _FallbackTfidfVectorizer:
    """Small TF-IDF vectorizer used when scikit-learn is unavailable."""

    def __init__(self) -> None:
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None

    def fit_transform(self, documents: list[str]) -> np.ndarray:
        self.fit(documents)
        return self.transform(documents)

    def fit(self, documents: list[str]) -> "_FallbackTfidfVectorizer":
        tokenized_docs = [self._tokenize(document) for document in documents]
        vocabulary = sorted({token for tokens in tokenized_docs for token in tokens})
        self.vocabulary_ = {token: index for index, token in enumerate(vocabulary)}

        doc_count = len(tokenized_docs)
        idf = np.ones(len(self.vocabulary_), dtype=float)

        for token, index in self.vocabulary_.items():
            document_frequency = sum(1 for tokens in tokenized_docs if token in set(tokens))
            idf[index] = math.log((1 + doc_count) / (1 + document_frequency)) + 1.0

        self.idf_ = idf
        return self

    def transform(self, documents: list[str]) -> np.ndarray:
        if self.idf_ is None:
            raise RuntimeError("Vectorizer must be fitted before calling transform().")

        matrix = np.zeros((len(documents), len(self.vocabulary_)), dtype=float)
        for row_index, document in enumerate(documents):
            tokens = self._tokenize(document)
            if not tokens:
                continue

            token_counts: dict[str, int] = {}
            for token in tokens:
                if token in self.vocabulary_:
                    token_counts[token] = token_counts.get(token, 0) + 1

            token_total = sum(token_counts.values())
            if token_total == 0:
                continue

            for token, count in token_counts.items():
                column_index = self.vocabulary_[token]
                term_frequency = count / token_total
                matrix[row_index, column_index] = term_frequency * self.idf_[column_index]

            row_norm = np.linalg.norm(matrix[row_index])
            if row_norm > 0:
                matrix[row_index] /= row_norm

        return matrix

    @staticmethod
    def _tokenize(document: str) -> list[str]:
        return document.split()


class Router:
    def __init__(
        self,
        min_similarity: float = 0.001,
        top_k: int = 10,
        lemma_language: str = "cs",
    ) -> None:
        self.min_similarity = min_similarity
        self.top_k = top_k
        self.lemma_language = lemma_language

        self.vectorizer: Any | None = None
        self.lemmatized_chunks: list[str] = []
        self.lemmatized_prompts: dict[str, str] = {}
        self.chunk_vectors: Any | None = None
        self.prompt_vectors: Any | None = None
        self.similarity_matrix: np.ndarray | None = None
        self.last_routes: dict[str, list[str]] = {}
        self.last_route_details: dict[str, list[dict[str, Any]]] = {}

    def route(
        self,
        chunks: list[str],
        parsers: list[object] | None = None,
    ) -> dict[str, list[str]]:
        parsers = parsers or self._default_parsers()
        parser_specs = self._build_parser_specs(parsers)
        if not chunks:
            self.last_routes = {spec["name"]: [] for spec in parser_specs}
            self.last_route_details = {spec["name"]: [] for spec in parser_specs}
            return self.last_routes

        lemmatized_chunks = [self._lemmatize_text(chunk) for chunk in chunks]
        vectorizer = self._build_vectorizer()
        chunk_vectors = vectorizer.fit_transform(lemmatized_chunks)

        parser_names = [spec["name"] for spec in parser_specs]
        lemmatized_prompts = [
            self._lemmatize_text(spec["prompt"])
            for spec in parser_specs
        ]
        prompt_vectors = vectorizer.transform(lemmatized_prompts)
        similarity_matrix = self._cosine_similarity(prompt_vectors, chunk_vectors)

        routes: dict[str, list[str]] = {}
        route_details: dict[str, list[dict[str, Any]]] = {}

        for parser_index, parser_name in enumerate(parser_names):
            scores = similarity_matrix[parser_index]
            selected_indices = self._select_chunk_indices(scores)

            routes[parser_name] = [chunks[index] for index in selected_indices]
            route_details[parser_name] = [
                {
                    "chunk_index": index,
                    "score": float(scores[index]),
                    "chunk": chunks[index],
                    "lemmatized_chunk": lemmatized_chunks[index],
                }
                for index in selected_indices
            ]

        self.vectorizer = vectorizer
        self.lemmatized_chunks = lemmatized_chunks
        self.lemmatized_prompts = dict(zip(parser_names, lemmatized_prompts, strict=False))
        self.chunk_vectors = chunk_vectors
        self.prompt_vectors = prompt_vectors
        self.similarity_matrix = similarity_matrix
        self.last_routes = routes
        self.last_route_details = route_details

        return routes

    def route_with_scores(
        self,
        chunks: list[str],
        parsers: list[object] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        self.route(chunks, parsers)
        return self.last_route_details

    def _build_vectorizer(self):
        if SklearnTfidfVectorizer is not None:
            return SklearnTfidfVectorizer(
                analyzer="word",
                lowercase=False,
                token_pattern=r"(?u)\b\w+\b",
                norm="l2",
            )
        return _FallbackTfidfVectorizer()

    def _build_parser_specs(self, parsers: list[object]) -> list[dict[str, str]]:
        return [
            {
                "name": self._parser_name(parser),
                "prompt": self._parser_prompt(parser),
            }
            for parser in parsers
        ]

    @staticmethod
    def _default_parsers() -> list[object]:
        return [
            LimitParser(gemini),
            DeductionParser(gemini),
            CoverageParser(gemini),
        ]

    def _parser_name(self, parser: object) -> str:
        return getattr(parser, "name", parser.__class__.__name__)

    def _parser_prompt(self, parser: object) -> str:
        prompt = getattr(parser, "routing_prompt", None) or getattr(parser, "system_prompt", None)
        if prompt:
            return prompt

        parser_name = parser.__class__.__name__
        prompt_by_parser_name = {
            "LimitParser": LIMIT_PARSER_SYSTEM_PROMT,
            "DeductionParser": DEDUCTION_PARSER_SYSTEM_PROMT,
            "CoverageParser": COVERAGE_PARSER_SYSTEM_PROMT,
        }

        if parser_name not in prompt_by_parser_name:
            raise ValueError(f"Unsupported parser type for routing: {parser_name}")

        return prompt_by_parser_name[parser_name]

    def _select_chunk_indices(self, scores: np.ndarray) -> list[int]:
        if scores.size == 0:
            return []

        ranked_indices = [int(index) for index in np.argsort(scores)[::-1]]
        threshold_matches = [
            index
            for index in ranked_indices
            if scores[index] >= self.min_similarity
        ]

        if threshold_matches:
            if self.top_k > 0:
                return threshold_matches[: self.top_k]
            return threshold_matches

        if self.top_k > 0:
            return ranked_indices[: self.top_k]

        return [ranked_indices[0]]

    def _lemmatize_text(self, text: str) -> str:
        lemmas = [self._lemmatize_token(token) for token in TOKEN_RE.findall(text)]
        return " ".join(lemma for lemma in lemmas if lemma)

    def _lemmatize_token(self, token: str) -> str:
        normalized = self._normalize_token(token)
        if not normalized:
            return ""

        if simplemma_lemmatize is not None:
            lemma = simplemma_lemmatize(normalized, lang=self.lemma_language)
            return self._normalize_token(str(lemma))

        return self._fallback_lemma(normalized)

    def _fallback_lemma(self, token: str) -> str:
        if len(token) <= 4:
            return token

        suffixes = (
            "kami",
            "emmi",
            "ovymi",
            "ovych",
            "ovich",
            "eho",
            "emu",
            "ami",
            "emi",
            "ove",
            "ovi",
            "ach",
            "ech",
            "ich",
            "ych",
            "ymi",
            "ami",
            "emi",
            "ate",
            "ove",
            "eni",
            "ani",
            "ost",
            "osti",
            "mi",
            "ou",
            "em",
            "om",
            "am",
            "um",
            "mi",
            "ho",
            "mu",
            "ch",
            "a",
            "e",
            "i",
            "y",
            "u",
            "o",
        )

        for suffix in suffixes:
            if token.endswith(suffix) and len(token) - len(suffix) >= 4:
                return token[: -len(suffix)]

        return token

    @staticmethod
    def _normalize_token(token: str) -> str:
        token = token.lower()
        token = unicodedata.normalize("NFKD", token)
        token = "".join(char for char in token if not unicodedata.combining(char))
        token = re.sub(r"[^0-9a-z]+", "", token)
        return token

    @staticmethod
    def _cosine_similarity(left_matrix: Any, right_matrix: Any) -> np.ndarray:
        left = np.asarray(left_matrix, dtype=float)
        right = np.asarray(right_matrix, dtype=float)

        if left.ndim == 1:
            left = left.reshape(1, -1)
        if right.ndim == 1:
            right = right.reshape(1, -1)

        left_norms = np.linalg.norm(left, axis=1, keepdims=True)
        right_norms = np.linalg.norm(right, axis=1, keepdims=True)

        left = np.divide(left, left_norms, out=np.zeros_like(left), where=left_norms != 0)
        right = np.divide(right, right_norms, out=np.zeros_like(right), where=right_norms != 0)

        return left @ right.T
