import argparse
import os
import gc
import torch
from transformers import (
    MarianTokenizer,
    MarianMTModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import pandas as pd
from huggingface_hub import login, create_repo, HfApi
from sklearn.model_selection import train_test_split
import sys
import threading
import queue
import matplotlib.pyplot as plt

class AsyncLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.log_queue = queue.Queue()
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._writer, daemon=True)
        self.thread.start()

    def write(self, msg):
        self.log_queue.put(msg)

    def flush(self):
        pass

    def _writer(self):
        with open(self.log_path, 'a', encoding='utf-8') as f:
            while not self._stop_event.is_set() or not self.log_queue.empty():
                try:
                    msg = self.log_queue.get(timeout=0.1)
                    f.write(msg)
                    f.flush()
                except queue.Empty:
                    continue

    def stop(self):
        self._stop_event.set()
        self.thread.join()

def parse_csv(file_path, src_lang, tgt_lang):
    df = pd.read_csv(file_path)
    if src_lang not in df.columns or tgt_lang not in df.columns:
        raise ValueError(f"CSV должен содержать колонки: '{src_lang}', '{tgt_lang}'")
    return [
        {"translation": {src_lang: row[src_lang], tgt_lang: row[tgt_lang]}}
        for _, row in df.iterrows()
        if isinstance(row[src_lang], str) and isinstance(row[tgt_lang], str)
    ]

def plot_loss_from_log(log_path, out_path='loss_plot.png'):
    import re
    import matplotlib.pyplot as plt
    steps, losses = [], []
    with open(log_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if "'loss':" in line:
                try:
                    loss = float(re.search(r"'loss': ([0-9.]+)", line).group(1))
                    losses.append(loss)
                    steps.append(i)
                except Exception:
                    continue
            elif "'eval_loss':" in line:
                try:
                    loss = float(re.search(r"'eval_loss': ([0-9.]+)", line).group(1))
                    losses.append(loss)
                    steps.append(i)
                except Exception:
                    continue
    if losses:
        plt.figure(figsize=(8,4))
        plt.plot(steps, losses, label='Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training/Eval Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        print(f"Loss plot saved to {out_path}")
    else:
        print("No loss values found in log.")

def main(args):
    # --- асинхронный логгер ---
    logger = AsyncLogger('train.log')
    sys.stdout = logger
    sys.stderr = logger

    login(token=args.token)
    src_lang, tgt_lang = ("en", "ru") if "en-ru" in args.model_name else ("ru", "en")
    # Разделение на train/validation
    translations = parse_csv(args.csv, src_lang, tgt_lang)
    train_data, val_data = train_test_split(translations, test_size=0.1, random_state=42)
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    tokenizer = MarianTokenizer.from_pretrained(args.model_name)

    def preprocess_function(examples):
        inputs = [ex[src_lang] for ex in examples['translation']]
        targets = [ex[tgt_lang] for ex in examples['translation']]
        model_inputs = tokenizer(inputs, max_length=args.max_length, truncation=True, padding="max_length")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=args.max_length, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=1000,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False
    )
    tokenized_val = val_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=1000,
        remove_columns=val_dataset.column_names,
        load_from_cache_file=False
    )
    del train_dataset, val_dataset
    gc.collect()

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=MarianMTModel.from_pretrained(args.model_name)
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        logging_steps=100,
        fp16=args.fp16 and torch.cuda.is_available(),
        push_to_hub=True,
        hub_model_id=args.repo_name,
        hub_token=args.token,
        report_to="none",
        save_safetensors=True,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=True,
        optim="adamw_torch",
        gradient_checkpointing=args.gradient_checkpointing,
        logging_dir="./logs",
        evaluation_strategy="epoch",
        predict_with_generate=False,
        load_best_model_at_end=True
    )

    model = MarianMTModel.from_pretrained(args.model_name)
    try:
        create_repo(args.repo_name, token=args.token, private=True, exist_ok=True)
    except Exception:
        pass

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    print(f"Обучение завершено! Final loss: {metrics['train_loss']:.4f}")
    trainer.save_model("final_model")
    trainer.push_to_hub(commit_message="Обучение завершено")
    # --- завершение логгера и построение графика ---
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    logger.stop()
    plot_loss_from_log('train.log')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Путь к CSV-файлу")
    parser.add_argument("--token", type=str, required=True, help="HuggingFace токен")
    parser.add_argument("--model_name", type=str, default="Helsinki-NLP/opus-mt-en-ru")
    parser.add_argument("--repo_name", type=str, default="fine-tuned-marian")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--dataloader_num_workers", type=int, default=2, help="Потоков загрузки данных (рекомендуется 4-8)")
    args = parser.parse_args()
    main(args)
