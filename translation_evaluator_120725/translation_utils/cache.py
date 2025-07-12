import json
import os
from typing import Dict, Any
from datetime import datetime, timedelta

class ResultsCache:
    """Кэш для хранения результатов оптимизации батчей."""
    
    def __init__(self, cache_file: str = "batch_optimization_cache.json"):
        self.cache_file = cache_file
        self.cache: Dict[str, Any] = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        """Загружает кэш из файла."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        """Сохраняет кэш в файл."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)

    def get_cached_result(self, key: str) -> Optional[Dict[str, Any]]:
        """Получает результат из кэша если он не устарел."""
        if key in self.cache:
            result = self.cache[key]
            cached_time = datetime.fromisoformat(result['timestamp'])
            if datetime.now() - cached_time < timedelta(hours=24):
                return result['data']
        return None

    def cache_result(self, key: str, data: Dict[str, Any]) -> None:
        """Сохраняет результат в кэш."""
        self.cache[key] = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        self._save_cache()