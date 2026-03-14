from typing import Self
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from models import Cache, session_context_var, transaction

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
        try:
            self.session.add(cache_entry)
            # Force INSERT inside this transaction scope to catch PK conflicts.
            self.session.flush()
            return True
        except IntegrityError:
            return False
