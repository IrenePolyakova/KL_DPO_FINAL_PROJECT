# API Documentation

## Model Loading (`model_loader.py`)

### `load_model_from_zip`
```python
def load_model_from_zip(
    zip_file: Union[str, IO],
    extract_path: str,
    device: Optional[str] = None,
    use_auth_token: bool = False
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Загружает модель из ZIP архива

    Args:
        zip_file: путь к ZIP файлу или файловый объект
        extract_path: путь для распаковки
        device: устройство для загрузки модели ('cuda' или 'cpu')
        use_auth_token: использовать ли токен для загрузки из Hugging Face Hub

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: модель и токенизатор
    """
```

## Translation (`translator.py`)

### `translate_texts`
```python
def translate_texts(
    texts: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    num_beams: int = 4,
    device: Optional[str] = None,
    show_progress: bool = False,
    n_jobs: int = -1
) -> Tuple[List[str], float]:
    """
    Параллельно переводит список текстов

    Args:
        texts: Список текстов для перевода
        model: Модель для перевода
        tokenizer: Токенизатор
        batch_size: Размер батча
        max_length: Максимальная длина последовательности
        num_beams: Количество лучей для beam search
        device: Устройство для вычислений ('cuda' или 'cpu')
        show_progress: Показывать ли прогресс-бар
        n_jobs: Количество процессов (-1 для использования всех ядер)

    Returns:
        Tuple[List[str], float]: список переводов и время выполнения
    """
```

### `translate_chunk`
```python
def translate_chunk(
    texts: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    num_beams: int = 4
) -> List[str]:
    """
    Переводит пакет текстов

    Args:
        texts: Список текстов для перевода
        model: Модель для перевода
        tokenizer: Токенизатор
        batch_size: Размер батча
        max_length: Максимальная длина последовательности
        num_beams: Количество лучей для beam search

    Returns:
        List[str]: список переводов
    """
```

## Metrics (`metrics.py`)

### `compute_corpus_metrics`
```python
def compute_corpus_metrics(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Вычисляет корпусные метрики качества перевода

    Args:
        predictions: Список переводов от модели
        references: Список эталонных переводов

    Returns:
        Dict[str, float]: словарь метрик {'bleu': float, 'chrf': float, 'ter': float}
    """
```

## File Processing

### DOCX Processing (`docx_utils.py`)
```python
def extract_content_from_docx(
    file: Union[str, IO]
) -> Dict[str, List[str]]:
    """
    Извлекает текст из DOCX файла

    Args:
        file: путь к файлу или файловый объект

    Returns:
        Dict[str, List[str]]: словарь с предложениями и таблицами
    """
```

### CSV Processing (`app.py`)
```python
def extract_content_from_csv(
    file: IO
) -> Dict[str, List[str]]:
    """
    Извлекает предложения из CSV файла

    Args:
        file: файловый объект

    Returns:
        Dict[str, List[str]]: словарь с предложениями и переводами
    """
```

### TMX Export (`tmx_utils.py`)
```python
def create_tmx_file(
    src_sentences: List[str],
    tgt_sentences: List[str],
    fileobj: Optional[Union[str, IO]] = None,
    sys_name: Optional[str] = None,
    source_lang: str = "en",
    target_lang: str = "ru"
) -> Union[str, bytes]:
    """
    Создает TMX файл с переводами

    Args:
        src_sentences: исходные предложения
        tgt_sentences: переведенные предложения
        fileobj: файловый объект или путь для сохранения
        sys_name: имя системы перевода
        source_lang: код исходного языка
        target_lang: код целевого языка

    Returns:
        Union[str, bytes]: путь к файлу или содержимое в виде bytes
    """
```

## Streamlit Interface (`app.py`)

### Main Application
```python
def main():
    """
    Основная функция приложения Streamlit
    """
```

### State Management
```python
def init_state():
    """
    Инициализирует состояние приложения
    """
```

### Results Display
```python
def display_results(
    df: pd.DataFrame,
    systems: List[str]
):
    """
    Отображает результаты сравнения

    Args:
        df: DataFrame с результатами
        systems: список систем перевода
    """
``` 