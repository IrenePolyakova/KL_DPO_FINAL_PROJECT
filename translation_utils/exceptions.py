from typing import Optional, Callable, Any
import time
from functools import wraps
import logging

logger = logging.getLogger(__name__)

class TranslationError(Exception):
    """Базовый класс для ошибок перевода"""
    pass

class ModelLoadError(TranslationError):
    """Ошибка загрузки модели"""
    pass

class TokenizationError(TranslationError):
    """Ошибка токенизации"""
    pass

class BatchProcessingError(TranslationError):
    """Ошибка обработки батча"""
    pass

class MemoryError(TranslationError):
    """Ошибка нехватки памяти"""
    pass

class CircuitBreaker:
    """
    Реализация паттерна Circuit Breaker для предотвращения каскадных отказов
    """
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: int = 60,
        half_open_timeout: int = 30
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_timeout = half_open_timeout
        
        self.failures = 0
        self.last_failure_time = 0
        self.state = "CLOSED"
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time >= self.reset_timeout:
                    self.state = "HALF-OPEN"
                    logger.info("Circuit Breaker переходит в состояние HALF-OPEN")
                else:
                    raise TranslationError("Circuit Breaker открыт, операция отклонена")
            
            try:
                result = func(*args, **kwargs)
                
                if self.state == "HALF-OPEN":
                    self.state = "CLOSED"
                    self.failures = 0
                    logger.info("Circuit Breaker успешно восстановлен")
                
                return result
                
            except Exception as e:
                self.failures += 1
                self.last_failure_time = time.time()
                
                if self.failures >= self.failure_threshold:
                    self.state = "OPEN"
                    logger.warning(
                        f"Circuit Breaker открыт после {self.failures} ошибок. "
                        f"Последняя ошибка: {str(e)}"
                    )
                
                raise
        
        return wrapper

class RetryWithBackoff:
    """
    Декоратор для повторных попыток с экспоненциальной задержкой
    """
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1,
        max_delay: float = 10,
        exceptions: tuple = (Exception,)
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exceptions = exceptions
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            retries = 0
            while retries < self.max_retries:
                try:
                    return func(*args, **kwargs)
                except self.exceptions as e:
                    retries += 1
                    if retries == self.max_retries:
                        raise
                    
                    delay = min(
                        self.base_delay * (2 ** (retries - 1)),
                        self.max_delay
                    )
                    
                    logger.warning(
                        f"Попытка {retries} не удалась: {str(e)}. "
                        f"Повторная попытка через {delay:.1f} сек."
                    )
                    
                    time.sleep(delay)
            
            return None  # Никогда не должны дойти сюда
        
        return wrapper 