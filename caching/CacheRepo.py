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