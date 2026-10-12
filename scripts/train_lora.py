"""QLoRA fine-tuning for a BUY/SELL/HOLD Qwen sequence classifier.

The JSONL produced by dataset_generator.py is read from
``data/jev_train_5m.jsonl`` by default. The resulting PEFT adapter is written
to ``models/qwen_adapter`` and can be loaded by backtest.py with
``--adapter-dir models/qwen_adapter`` (the default).

Run on a CUDA GPU pod with recent torch, transformers, datasets, peft,
accelerate, and bitsandbytes installed.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LABEL2ID = {"BUY": 0, "SELL": 1, "HOLD": 2}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/jev_train_5m.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Qwen/Qwen2.5-3B-Instruct or Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-dir", default="models/qwen_adapter")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    global torch, load_dataset, LoraConfig, TaskType, get_peft_model
    global prepare_model_for_kbit_training, AutoModelForSequenceClassification
    global AutoTokenizer, BitsAndBytesConfig, DataCollatorWithPadding, Trainer, TrainingArguments

    args = parse_args()
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    if args.max_length < 8 or args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("max_length, batch_size, and gradient accumulation must be positive")
    if not 0 <= args.eval_ratio < 1:
        raise ValueError("eval_ratio must be in [0, 1)")
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    args.data = str(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Training JSONL not found: {data_path}")
    output_path = Path(args.output_dir)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    args.output_dir = str(output_path)

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires a CUDA GPU; run this script on a GPU pod")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = load_dataset("json", data_files=str(data_path), split="train")
    required = {"prompt", "action"}
    if not required.issubset(dataset.column_names):
        raise ValueError(f"JSONL needs fields {sorted(required)}")
    if "meta" not in dataset.column_names:
        raise ValueError("JSONL needs meta barrier metadata from dataset_generator.py")

    # Fail fast on legacy or incompatible training files: prompts need the HTF
    # regime feature and labels must carry the same asymmetric barriers used by
    # the current strategy/backtest.
    for row in dataset.select(range(min(len(dataset), 100))):
        if "htf_ema=" not in row["prompt"]:
            raise ValueError("Training prompts lack the htf_ema feature; regenerate with dataset_generator.py")
        meta = row["meta"]
        if (abs(float(meta.get("stop_atr_multiplier", -1)) - 1.0) > 1e-9
                or abs(float(meta.get("target_atr_multiplier", -1)) - 2.2) > 1e-9):
            raise ValueError("Training data must use 1.0x ATR stops and 2.2x ATR targets")

    def encode_batch(batch):
        enc = tokenizer(batch["prompt"], truncation=True, max_length=args.max_length)
        enc["labels"] = [LABEL2ID.get(str(label).upper(), -1) for label in batch["action"]]
        return enc

    dataset = dataset.map(encode_batch, batched=True, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda row: row["labels"] >= 0)
    if len(dataset) < 10:
        raise ValueError("Need at least 10 valid labeled rows to train and evaluate")
    split = dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed) if args.eval_ratio else None
    train_dataset = split["train"] if split else dataset
    eval_dataset = split["test"] if split else None

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABEL2ID),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        quantization_config=quantization,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["score"],
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    training_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )
    signature = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in signature else "evaluation_strategy"
    training_kwargs[eval_key] = "epoch" if eval_dataset is not None else "no"
    if eval_dataset is not None:
        training_kwargs["load_best_model_at_end"] = True
        training_kwargs["metric_for_best_model"] = "eval_loss"
        training_kwargs["greater_is_better"] = False
    training_args = TrainingArguments(**training_kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        processing_class=tokenizer,
    )
    trainer.train()
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    print(f"Saved Qwen sequence-classification LoRA adapter to {output.resolve()}")


if __name__ == "__main__":
    main()
