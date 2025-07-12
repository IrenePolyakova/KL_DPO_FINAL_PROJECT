from sacrebleu.metrics import BLEU, CHRF, TER

# Создаём экземпляры метрик один раз для повторного использования
bleu_metric = BLEU()
chrf_metric = CHRF()
ter_metric = TER()

def compute_metrics(pred, ref):
    """Вычисляет метрики качества перевода для одного предложения"""
    bleu = bleu_metric.sentence_score(pred, [ref]).score / 100
    chrf = chrf_metric.sentence_score(pred, [ref]).score / 100
    ter = ter_metric.sentence_score(pred, [ref]).score / 100

    return {
        "bleu": bleu,
        "chrf": chrf,
        "ter": ter
    }
