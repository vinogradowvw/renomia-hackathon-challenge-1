from typing import Self
from sqlalchemy import bindparam, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from models import Cache, session_context_var

class CacheRepo:
    """Repository for caching LLM responses."""
    _instance = None
    
    @classmethod
    def get_instance(cls) -> Self:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @property
    def session(self) -> Session:
        return session_context_var.get()
    
    def get_by_key(self, key: str) -> Cache | None:
        """Retrieve a cached response for a given query."""
        return self.session.query(Cache).filter_by(key=key).first()
    
    def if_exists(self, key: str) -> bool:
        """Check if a cached response exists for a given query."""
        return self.session.query(Cache).filter_by(key=key).first() is not None
    
    def add(self, cache_entry: Cache) -> bool:
        """Save a new response to the cache."""
        if self.if_exists(cache_entry.key):
            return False
        try:
            # Isolate conflicts in a savepoint so outer transaction can continue.
            with self.session.begin_nested():
                self.session.add(cache_entry)
                # Force INSERT now to detect PK conflict in this method.
                self.session.flush()
            return True
        except IntegrityError as err:
            pg_code = getattr(err.orig, "pgcode", None) or getattr(err.orig, "sqlstate", None)
            if pg_code == "23505":
                return False
            raise

    def rank_offer_ids(self, keys: list[str]) -> list[str]:
        """Rank offers from cache by max-field wins and tie-breaker priority."""
        if not keys:
            return []

        stmt = text(
            """
            WITH batch AS (
                SELECT
                    c.key,
                    c.value,
                    COALESCE(c.value->>'id', c.key) AS offer_id,
                    COALESCE((
                        SELECT COUNT(*)
                        FROM unnest(string_to_array(COALESCE(c.value->>'territorial_scope', ''), ',')) AS p(country)
                        WHERE btrim(p.country) <> ''
                    ), 0) AS territorial_count,
                    NULLIF(c.value->>'basic_limit_czk', '')::numeric AS basic_limit_czk,
                    NULLIF(c.value->>'limit_multiplier_per_year', '')::numeric AS limit_multiplier_per_year,
                    NULLIF(c.value->>'aggregate_limit_czk', '')::numeric AS aggregate_limit_czk,
                    NULLIF(c.value->>'limit_persons_in_custody_czk', '')::numeric AS limit_persons_in_custody_czk,
                    NULLIF(c.value->>'limit_pure_financial_loss_czk', '')::numeric AS limit_pure_financial_loss_czk,
                    NULLIF(c.value->>'limit_taken_items_czk', '')::numeric AS limit_taken_items_czk,
                    NULLIF(c.value->>'limit_cross_liability_czk', '')::numeric AS limit_cross_liability_czk,
                    NULLIF(c.value->>'limit_recourse_czk', '')::numeric AS limit_recourse_czk,
                    NULLIF(c.value->>'limit_non_pecuniary_damage_czk', '')::numeric AS limit_non_pecuniary_damage_czk,
                    NULLIF(c.value->>'basic_deductible_czk', '')::numeric AS basic_deductible_czk,
                    NULLIF(c.value->>'deductible_recourse_czk', '')::numeric AS deductible_recourse_czk,
                    NULLIF(c.value->>'deductible_non_pecuniary_czk', '')::numeric AS deductible_non_pecuniary_czk,
                    NULLIF(c.value->>'deductible_brought_items_czk', '')::numeric AS deductible_brought_items_czk,
                    NULLIF(c.value->>'deductible_financial_loss_czk', '')::numeric AS deductible_financial_loss_czk,
                    NULLIF(c.value->>'premium_czk', '')::numeric AS premium_czk
                FROM cache c
                WHERE c.key IN :keys
            ),
            mx AS (
                SELECT
                    MAX(territorial_count) AS territorial_count,
                    MAX(basic_limit_czk) AS basic_limit_czk,
                    MAX(limit_multiplier_per_year) AS limit_multiplier_per_year,
                    MAX(aggregate_limit_czk) AS aggregate_limit_czk,
                    MAX(limit_persons_in_custody_czk) AS limit_persons_in_custody_czk,
                    MAX(limit_pure_financial_loss_czk) AS limit_pure_financial_loss_czk,
                    MAX(limit_taken_items_czk) AS limit_taken_items_czk,
                    MAX(limit_cross_liability_czk) AS limit_cross_liability_czk,
                    MAX(limit_recourse_czk) AS limit_recourse_czk,
                    MAX(limit_non_pecuniary_damage_czk) AS limit_non_pecuniary_damage_czk,
                    MAX(basic_deductible_czk) AS basic_deductible_czk,
                    MAX(deductible_recourse_czk) AS deductible_recourse_czk,
                    MAX(deductible_non_pecuniary_czk) AS deductible_non_pecuniary_czk,
                    MAX(deductible_brought_items_czk) AS deductible_brought_items_czk,
                    MAX(deductible_financial_loss_czk) AS deductible_financial_loss_czk,
                    MAX(premium_czk) AS premium_czk
                FROM batch
            ),
            scored AS (
                SELECT
                    b.offer_id,
                    (
                        (CASE WHEN b.territorial_count = mx.territorial_count THEN 1 ELSE 0 END) +
                        (CASE WHEN b.basic_limit_czk = mx.basic_limit_czk THEN 1 ELSE 0 END) +
                        (CASE WHEN b.limit_multiplier_per_year = mx.limit_multiplier_per_year THEN 1 ELSE 0 END) +
                        (CASE WHEN b.aggregate_limit_czk = mx.aggregate_limit_czk THEN 1 ELSE 0 END) +
                        (CASE WHEN b.limit_persons_in_custody_czk = mx.limit_persons_in_custody_czk THEN 1 ELSE 0 END) +
                        (CASE WHEN b.limit_pure_financial_loss_czk = mx.limit_pure_financial_loss_czk THEN 1 ELSE 0 END) +
                        (CASE WHEN b.limit_taken_items_czk = mx.limit_taken_items_czk THEN 1 ELSE 0 END) +
                        (CASE WHEN b.limit_cross_liability_czk = mx.limit_cross_liability_czk THEN 1 ELSE 0 END) +
                        (CASE WHEN b.limit_recourse_czk = mx.limit_recourse_czk THEN 1 ELSE 0 END) +
                        (CASE WHEN b.limit_non_pecuniary_damage_czk = mx.limit_non_pecuniary_damage_czk THEN 1 ELSE 0 END) +
                        (CASE WHEN b.basic_deductible_czk = mx.basic_deductible_czk THEN 1 ELSE 0 END) +
                        (CASE WHEN b.deductible_recourse_czk = mx.deductible_recourse_czk THEN 1 ELSE 0 END) +
                        (CASE WHEN b.deductible_non_pecuniary_czk = mx.deductible_non_pecuniary_czk THEN 1 ELSE 0 END) +
                        (CASE WHEN b.deductible_brought_items_czk = mx.deductible_brought_items_czk THEN 1 ELSE 0 END) +
                        (CASE WHEN b.deductible_financial_loss_czk = mx.deductible_financial_loss_czk THEN 1 ELSE 0 END) +
                        (CASE WHEN b.premium_czk = mx.premium_czk THEN 1 ELSE 0 END)
                    ) AS max_count,
                    b.territorial_count,
                    b.basic_limit_czk,
                    b.limit_multiplier_per_year,
                    b.aggregate_limit_czk,
                    b.limit_persons_in_custody_czk,
                    b.limit_pure_financial_loss_czk,
                    b.limit_taken_items_czk,
                    b.limit_cross_liability_czk,
                    b.limit_recourse_czk,
                    b.limit_non_pecuniary_damage_czk,
                    b.basic_deductible_czk,
                    b.deductible_recourse_czk,
                    b.deductible_non_pecuniary_czk,
                    b.deductible_brought_items_czk,
                    b.deductible_financial_loss_czk,
                    b.premium_czk
                FROM batch b
                CROSS JOIN mx
            )
            SELECT offer_id
            FROM scored
            ORDER BY
                max_count DESC,
                territorial_count DESC,
                basic_limit_czk DESC NULLS LAST,
                limit_multiplier_per_year DESC NULLS LAST,
                aggregate_limit_czk DESC NULLS LAST,
                limit_persons_in_custody_czk DESC NULLS LAST,
                limit_pure_financial_loss_czk DESC NULLS LAST,
                limit_taken_items_czk DESC NULLS LAST,
                limit_cross_liability_czk DESC NULLS LAST,
                limit_recourse_czk DESC NULLS LAST,
                limit_non_pecuniary_damage_czk DESC NULLS LAST,
                basic_deductible_czk DESC NULLS LAST,
                deductible_recourse_czk DESC NULLS LAST,
                deductible_non_pecuniary_czk DESC NULLS LAST,
                deductible_brought_items_czk DESC NULLS LAST,
                deductible_financial_loss_czk DESC NULLS LAST,
                premium_czk DESC NULLS LAST
            """
        ).bindparams(bindparam("keys", expanding=True))

        rows = self.session.execute(stmt, {"keys": keys}).all()
        return [row.offer_id for row in rows]
