from sacrebleu.metrics import BLEU, CHRF, TER
from typing import List, Dict, Union, Optional, Tuple
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import logging
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import nltk
import sacrebleu
from collections import defaultdict

try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception:
    pass

logger = logging.getLogger(__name__)

# Создаём экземпляры метрик один раз для повторного использования
bleu_metric = BLEU(effective_order=True)  # Используем эффективный порядок для коротких предложений
chrf_metric = CHRF(word_order=2)  # Включаем учет порядка слов
ter_metric = TER(normalized=True)  # Используем нормализованную версию

# Загружаем модель для оценки качества
try:
    comet_model = None
    comet_tokenizer = None
    def load_comet():
        global comet_model, comet_tokenizer
        if comet_model is None:
            model_name = "Unbabel/wmt22-comet-da"
            comet_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            comet_tokenizer = AutoTokenizer.from_pretrained(model_name)
            if torch.cuda.is_available():
                comet_model = comet_model.cuda()
            comet_model.eval()
except Exception as e:
    logger.warning(f"Не удалось загрузить COMET модель: {str(e)}")

def compute_comet_score(src: str, pred: str, ref: str) -> float:
    """Вычисляет COMET score для одного предложения"""
    try:
        if comet_model is None:
            load_comet()
        
        inputs = comet_tokenizer(
            [src], [pred], [ref],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = comet_model(**inputs)
            score = outputs.logits.squeeze().cpu().item()
        
        return (score + 1) / 2  # Нормализуем к [0, 1]
    except Exception as e:
        logger.warning(f"Ошибка при вычислении COMET score: {str(e)}")
        return 0.0

def compute_sentence_metrics(
    pred: str,
    ref: str,
    src: Optional[str] = None,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Вычисляет метрики качества перевода для одного предложения
    
    Args:
        pred: Предсказанный перевод
        ref: Эталонный перевод
        src: Исходное предложение (для COMET)
        metrics: Список метрик для вычисления (по умолчанию все)
        
    Returns:
        Dict[str, float]: Словарь с метриками
    """
    if not pred or not ref:
        return {
            "bleu": 0.0,
            "chrf": 0.0,
            "ter": 0.0,
            "meteor": 0.0,
            "comet": 0.0,
            "length_ratio": 0.0
        }
    
    if metrics is None:
        metrics = ["bleu", "chrf", "ter", "meteor", "comet", "length_ratio"]
    
    result = {}
    
    try:
        if "bleu" in metrics:
            result["bleu"] = bleu_metric.sentence_score(pred, [ref]).score / 100
            
        if "chrf" in metrics:
            result["chrf"] = chrf_metric.sentence_score(pred, [ref]).score / 100
            
        if "ter" in metrics:
            result["ter"] = ter_metric.sentence_score(pred, [ref]).score / 100
            
        if "meteor" in metrics:
            # Токенизируем и вычисляем METEOR
            pred_tokens = word_tokenize(pred.lower())
            ref_tokens = word_tokenize(ref.lower())
            result["meteor"] = meteor_score([ref_tokens], pred_tokens)
            
        if "comet" in metrics and src is not None:
            result["comet"] = compute_comet_score(src, pred, ref)
            
        if "length_ratio" in metrics:
            pred_len = len(pred.split())
            ref_len = len(ref.split())
            result["length_ratio"] = pred_len / ref_len if ref_len > 0 else 0.0
            
    except Exception as e:
        logger.warning(f"Ошибка при вычислении метрик: {str(e)}")
        # Возвращаем нули при ошибке
        result = {metric: 0.0 for metric in metrics}
    
    return result

def compute_corpus_metrics(
    preds: List[str],
    refs: List[str],
    srcs: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    n_jobs: int = -1
) -> Dict[str, float]:
    """
    Вычисляет метрики качества перевода для корпуса
    
    Args:
        preds: Список предсказанных переводов
        refs: Список эталонных переводов
        srcs: Список исходных предложений (для COMET)
        metrics: Список метрик для вычисления
        n_jobs: Количество процессов (-1 для использования всех ядер)
        
    Returns:
        Dict[str, float]: Словарь с метриками
    """
    if not preds or not refs or len(preds) != len(refs):
        return {
            "bleu": 0.0,
            "chrf": 0.0,
            "ter": 0.0,
            "meteor": 0.0,
            "comet": 0.0,
            "length_ratio": 0.0
        }
    
    if metrics is None:
        metrics = ["bleu", "chrf", "ter", "meteor", "comet", "length_ratio"]
    
    try:
        # Корпусные метрики
        result = {}
        
        if "bleu" in metrics:
            result["bleu"] = bleu_metric.corpus_score(preds, [refs]).score / 100
            
        if "chrf" in metrics:
            result["chrf"] = chrf_metric.corpus_score(preds, [refs]).score / 100
            
        if "ter" in metrics:
            result["ter"] = ter_metric.corpus_score(preds, [refs]).score / 100
            
        if "meteor" in metrics:
            # Вычисляем METEOR для каждого предложения и усредняем
            meteor_scores = []
            for pred, ref in zip(preds, refs):
                pred_tokens = word_tokenize(pred.lower())
                ref_tokens = word_tokenize(ref.lower())
                score = meteor_score([ref_tokens], pred_tokens)
                meteor_scores.append(score)
            result["meteor"] = np.mean(meteor_scores)
            
        if "comet" in metrics and srcs is not None:
            # Вычисляем COMET для каждого предложения и усредняем
            comet_scores = []
            for src, pred, ref in zip(srcs, preds, refs):
                score = compute_comet_score(src, pred, ref)
                comet_scores.append(score)
            result["comet"] = np.mean(comet_scores)
            
        if "length_ratio" in metrics:
            # Вычисляем отношение длин для всего корпуса
            pred_lens = [len(pred.split()) for pred in preds]
            ref_lens = [len(ref.split()) for ref in refs]
            result["length_ratio"] = np.mean(pred_lens) / np.mean(ref_lens) if np.mean(ref_lens) > 0 else 0.0
            
    except Exception as e:
        logger.warning(f"Ошибка при вычислении корпусных метрик: {str(e)}")
        # Возвращаем нули при ошибке
        result = {metric: 0.0 for metric in metrics}
    
    return result

def compute_metrics_batch(
    preds: List[str],
    refs: List[str],
    srcs: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    n_jobs: int = -1,
    batch_size: int = 32
) -> List[Dict[str, float]]:
    """
    Вычисляет метрики для пакета предложений параллельно
    
    Args:
        preds: Список предсказанных переводов
        refs: Список эталонных переводов
        srcs: Список исходных предложений (для COMET)
        metrics: Список метрик для вычисления
        n_jobs: Количество процессов (-1 для использования всех ядер)
        batch_size: Размер пакета для COMET
        
    Returns:
        List[Dict[str, float]]: Список словарей с метриками для каждого предложения
    """
    if not preds or not refs or len(preds) != len(refs):
        return [
            {
                "bleu": 0.0, "chrf": 0.0, "ter": 0.0,
                "meteor": 0.0, "comet": 0.0,
                "length_ratio": 0.0
            }
            for _ in range(max(len(preds), len(refs)))
        ]
    
    # Создаем частичную функцию с фиксированными метриками
    if srcs:
        compute_fn = partial(compute_sentence_metrics, metrics=metrics)
        data = list(zip(preds, refs, srcs))
    else:
        compute_fn = partial(compute_sentence_metrics, src=None, metrics=metrics)
        data = list(zip(preds, refs))
    
    try:
        # Разбиваем на пакеты для оптимизации COMET
        results = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            
            # Параллельное вычисление метрик
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                if srcs:
                    batch_results = list(executor.map(
                        lambda x: compute_fn(x[0], x[1], x[2]),
                        batch
                    ))
                else:
                    batch_results = list(executor.map(
                        lambda x: compute_fn(x[0], x[1]),
                        batch
                    ))
                results.extend(batch_results)
        
        return results
        
    except Exception as e:
        logger.error(f"Ошибка при пакетном вычислении метрик: {str(e)}")
        return [
            {
                "bleu": 0.0, "chrf": 0.0, "ter": 0.0,
                "meteor": 0.0, "comet": 0.0,
                "length_ratio": 0.0
            }
            for _ in range(len(preds))
        ]

def get_metric_weights() -> Dict[str, float]:
    """Возвращает веса для каждой метрики"""
    return {
        "bleu": 0.2,
        "chrf": 0.2,
        "ter": 0.1,
        "meteor": 0.2,
        "comet": 0.3
    }

def compute_weighted_score(metrics: Dict[str, float]) -> float:
    """Вычисляет взвешенную оценку качества перевода"""
    weights = get_metric_weights()
    score = 0.0
    weight_sum = 0.0
    
    for metric, value in metrics.items():
        if metric in weights and not np.isnan(value):
            score += weights[metric] * value
            weight_sum += weights[metric]
    
    return score / weight_sum if weight_sum > 0 else 0.0

def compute_metrics(
    hypothesis: str,
    reference: str,
) -> Dict[str, float]:
    """
    Вычисляет метрики для одной пары перевод-эталон.
    
    Args:
        hypothesis: Предполагаемый перевод
        reference: Эталонный перевод
        
    Returns:
        Dict[str, float]: Словарь с метриками
    """
    try:
        # Обрабатываем пустые строки
        if not hypothesis or not reference:
            return {
                'bleu': 0.0,
                'chrf': 0.0,
                'ter': 0.0
            }
        
        # Вычисляем BLEU
        bleu = sacrebleu.sentence_bleu(hypothesis, [reference]).score
        
        # Вычисляем chrF
        chrf = sacrebleu.sentence_chrf(hypothesis, [reference]).score
        
        # Вычисляем TER
        ter = sacrebleu.sentence_ter(hypothesis, [reference]).score
        
        return {
            'bleu': round(bleu, 2),
            'chrf': round(chrf, 2),
            'ter': round(ter, 2)
        }
        
    except Exception as e:
        logger.error(f"Ошибка при вычислении метрик: {str(e)}")
        return {
            'bleu': 0.0,
            'chrf': 0.0,
            'ter': 0.0
        }



def aggregate_metrics(
    metrics_list: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    Агрегирует список метрик в средние значения.
    
    Args:
        metrics_list: Список словарей с метриками
        
    Returns:
        Dict[str, float]: Словарь со средними значениями метрик
    """
    try:
        if not metrics_list:
            return {
                'bleu': 0.0,
                'chrf': 0.0,
                'ter': 0.0
            }
        
        # Собираем все значения для каждой метрики
        aggregated = defaultdict(list)
        for metrics in metrics_list:
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    aggregated[metric].append(value)
        
        # Вычисляем средние значения
        return {
            metric: round(float(np.mean(values)), 2)
            for metric, values in aggregated.items()
        }
        
    except Exception as e:
        logger.error(f"Ошибка при агрегации метрик: {str(e)}")
        return {
            'bleu': 0.0,
            'chrf': 0.0,
            'ter': 0.0
        }
