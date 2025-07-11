import torch
import psutil
import time
from typing import Optional, Tuple, Dict, Any, List
from functools import lru_cache
from .models import BatchConfig, ModelConfig
from .exceptions import BatchProcessingError
import logging
from tqdm import tqdm
from .cache import ResultsCache

logger = logging.getLogger(__name__)

class BatchOptimizer:
    """
    Оптимизатор размера батча для нейронных моделей перевода.
    
    Attributes:
        model_config (ModelConfig): Конфигурация модели
        batch_config (BatchConfig): Конфигурация батча
        optimal_batch_size (Optional[int]): Оптимальный размер батча
        performance_history (List[Dict[str, float]]): История производительности
    """

    def __init__(self, model_config: ModelConfig, batch_config: BatchConfig):
        """
        Args:
            model_config: Конфигурация модели
            batch_config: Конфигурация батча
        
        Raises:
            ValueError: Если параметры конфигурации невалидны
        """
        self._validate_configs(model_config, batch_config)
        self.model_config = model_config
        self.batch_config = batch_config
        self.optimal_batch_size: Optional[int] = None
        self.performance_history: List[Dict[str, float]] = []
        self.cache = ResultsCache()

    @staticmethod
    def _validate_configs(model_config: ModelConfig, batch_config: BatchConfig) -> None:
        """Проверяет валидность конфигураций."""
        if not model_config.model or not model_config.tokenizer:
            raise ValueError("Model and tokenizer must be provided")
        if batch_config.min_batch_size < 1:
            raise ValueError("Minimum batch size must be at least 1")
        if batch_config.max_batch_size < batch_config.min_batch_size:
            raise ValueError("Maximum batch size must be greater than minimum")
        if batch_config.memory_limit <= 0:
            raise ValueError("Memory limit must be positive")

    @lru_cache(maxsize=32)
    def measure_batch_performance(self, batch_size: int, sample_text: str) -> Tuple[float, float]:
        """
        Измеряет производительность для заданного размера батча.
        
        Args:
            batch_size: Размер батча для тестирования
            sample_text: Пример текста для тестирования
        
        Returns:
            Tuple[float, float]: (время выполнения, использованная память)
        
        Raises:
            BatchProcessingError: При ошибке обработки батча
        """
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Batch size must be a positive integer")
        if not sample_text or not isinstance(sample_text, str):
            raise ValueError("Sample text must be a non-empty string")

        try:
            batch = [sample_text] * batch_size
            start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            start_time = time.time()

            with torch.no_grad():
                _ = self.model_config.model.generate(
                    input_ids=self.model_config.tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.model_config.max_length
                    ).input_ids.to(self.model_config.device)
                )

            end_time = time.time()
            end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            # Сохраняем метрики в историю
            self.performance_history.append({
                'batch_size': batch_size,
                'time_taken': end_time - start_time,
                'memory_used': end_memory - start_memory,
                'timestamp': time.time()
            })
            
            return end_time - start_time, end_memory - start_memory

        except RuntimeError as e:
            if "out of memory" in str(e):
                return float('inf'), float('inf')
            raise BatchProcessingError(f"Error processing batch: {str(e)}")

    def find_optimal_batch_size(self, sample_text: str) -> int:
        """
        Находит оптимальный размер батча с помощью бинарного поиска.
        
        Args:
            sample_text: Текст для тестирования производительности
            
        Returns:
            int: Оптимальный размер батча
            
        Raises:
            BatchProcessingError: При ошибке оптимизации
        """
        logger.info("Начало поиска оптимального размера батча")
        
        # Проверяем кэш
        cache_key = f"{self.model_config.model.__class__.__name__}_{len(sample_text)}"
        cached_result = self.cache.get_cached_result(cache_key)
        if cached_result:
            self.optimal_batch_size = cached_result['optimal_batch_size']
            return self.optimal_batch_size

        left = self.batch_config.min_batch_size
        right = self.batch_config.max_batch_size
        optimal_batch_size = left
        best_throughput = 0

        while left <= right:
            batch_size = (left + right) // 2
            try:
                time_taken, memory_used = self.measure_batch_performance(batch_size, sample_text)
                
                if memory_used > self.batch_config.memory_limit:
                    right = batch_size - 1
                    continue

                throughput = batch_size / time_taken
                logger.info(f"Batch size {batch_size}: throughput={throughput:.2f} items/sec, "
                          f"memory={memory_used/1024/1024:.2f}MB")

                if throughput > best_throughput:
                    best_throughput = throughput
                    optimal_batch_size = batch_size
                    left = batch_size + 1
                else:
                    right = batch_size - 1

            except RuntimeError as e:
                if "out of memory" in str(e):
                    right = batch_size - 1
                else:
                    raise

        self.optimal_batch_size = optimal_batch_size
        logger.info(f"Найден оптимальный размер батча: {optimal_batch_size}")

        # Сохраняем результат в кэш
        self.cache.cache_result(cache_key, {
            'optimal_batch_size': self.optimal_batch_size,
            'performance_history': self.performance_history
        })
        
        return self.optimal_batch_size

class ProgressTracker:
    def __init__(self, total_items: int, desc: str = "Processing"):
        self.total = total_items
        self.desc = desc
        self.start_time = time.time()
        self.pbar = tqdm(total=total_items, desc=desc)
        self.processed = 0
        self.metrics = {
            'avg_time_per_item': 0,
            'estimated_time_remaining': 0,
            'memory_usage': 0,
        }

    def update(self, items_processed: int = 1):
        self.processed += items_processed
        self.pbar.update(items_processed)
        
        # Обновляем метрики
        elapsed_time = time.time() - self.start_time
        self.metrics['avg_time_per_item'] = elapsed_time / self.processed
        self.metrics['estimated_time_remaining'] = (
            self.metrics['avg_time_per_item'] * (self.total - self.processed)
        )
        self.metrics['memory_usage'] = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Обновляем описание с метриками
        self.pbar.set_postfix({
            'avg_time': f"{self.metrics['avg_time_per_item']:.2f}s",
            'eta': f"{self.metrics['estimated_time_remaining']:.0f}s",
            'memory': f"{self.metrics['memory_usage']:.0f}MB"
        })

    def close(self):
        self.pbar.close()
        return self.metrics