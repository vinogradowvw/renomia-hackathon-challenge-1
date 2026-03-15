import hashlib
import unicodedata


class Hashing:
    @staticmethod
    def normalize_text(text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text or "").lower()
        return "".join(ch for ch in normalized if ch.isalnum())

    @classmethod
    def sha256(cls, text: str) -> str:
        prepared = cls.normalize_text(text)
        return hashlib.sha256(prepared.encode("utf-8")).hexdigest()

