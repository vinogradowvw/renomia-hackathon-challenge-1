from caching.CacheRepo import CacheRepo
from models import transaction

class Ranking:
    
    def rank(self, keys: list[str]) -> list[str]:
        """Rank cached offers using repository query logic."""
        with transaction():
            return CacheRepo.get_instance().rank_offer_ids(keys)
    
    
