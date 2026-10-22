"""QLoRA fine-tuning for a BUY/SELL/HOLD Qwen sequence classifier.

The JSONL produced by dataset_generator.py is read from
``data/jev_train_5m.jsonl`` by default. The resulting PEFT adapter is written
to ``models/qwen_adapter`` and can be loaded by backtest.py with
``--adapter-dir models/qwen_adapter`` (the default).

The per-example loss is R-weighted: directional rows that realized more
multiples of their stop (``meta.realized_r``) teach proportionally more, and
background no-setup HOLD rows are down-weighted so the 99% HOLD majority
cannot swamp the directional signal the classifier exists to learn. Pass
``--r-weight-scale 0 --hold-weight 1.0`` to recover plain cross-entropy.

Run on a CUDA GPU pod with recent torch, transformers, datasets, peft,
accelerate, and bitsandbytes installed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import inspect
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LABEL2ID = {"BUY": 0, "SELL": 1, "HOLD": 2}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DIRECTIONAL = {"BUY", "SELL"}


def example_weight(
    action: str,
    meta: dict,
    r_weight_scale: float = 1.0,
    hold_weight: float = 0.25,
) -> float:
    """Per-example loss weight from the label and the barrier outcome.

    - Directional label with a realized outcome: ``1 + r_weight_scale * R``
      (a +2.2R winner teaches hardest; scales linearly, 0 disables).
    - HOLD from a gated candidate (stop/timeout): full weight 1.0 -- these
      are the false positives of the rules and carry the selectivity signal.
    - HOLD with no candidate (the vast background): ``hold_weight``.
    - Legacy rows without ``realized_r`` fall back to 1.0 / ``hold_weight``.
    """
    action = str(action).upper()
    realized = meta.get("realized_r") if isinstance(meta, dict) else None
    if realized is not None:
        if action in DIRECTIONAL:
            return 1.0 + r_weight_scale * max(float(realized), 0.0)
        return 1.0
    if action in DIRECTIONAL:
        return 1.0
    return hold_weight


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
    parser.add_argument("--purge-bars", type=int, default=0,
                        help="Extra chronological gap before validation; actual purge is at least the label horizon")
    parser.add_argument("--r-weight-scale", type=float, default=1.0,
                        help="Loss weight slope on realized R for directional rows (0 disables R-weighting)")
    parser.add_argument("--hold-weight", type=float, default=0.25,
                        help="Loss weight for background no-setup HOLD rows (1.0 disables rebalancing)")
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
    if not 0 < args.eval_ratio < 1:
        raise ValueError("eval_ratio must be > 0 and < 1 to report validation metrics")
    if args.purge_bars < 0:
        raise ValueError("purge_bars cannot be negative")
    if args.r_weight_scale < 0:
        raise ValueError("r_weight_scale cannot be negative")
    if not 0 < args.hold_weight <= 1:
        raise ValueError("hold_weight must be in (0, 1]")
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

    if len(dataset) < 20:
        raise ValueError("Need at least 20 valid labeled rows for a chronological train/validation split")

    # Validate every record. Directional labels must attest to a target-first
    # outcome; HOLD remains the label for stops, timeouts, and blocked regimes.
    previous_bar_index = -1
    previous_timestamp = None
    timestamp_flags: set[bool] = set()
    maximum_horizon = 0
    maximum_horizon_minutes = 0
    maximum_bar_minutes = 0
    for row in dataset:
        if "htf_ema=" not in row["prompt"]:
            raise ValueError("Training prompts lack the htf_ema feature; regenerate with dataset_generator.py")
        for required_prompt_text in (
            "minimum 2.2x ATR profit target",
            "matching rule signal",
            "choose HOLD",
        ):
            if required_prompt_text not in row["prompt"]:
                raise ValueError("Training prompt lacks the current strategy objective; regenerate the JSONL")
        meta = row["meta"]
        if (abs(float(meta.get("stop_atr_multiplier", -1)) - 1.0) > 1e-9
                or abs(float(meta.get("target_atr_multiplier", -1)) - 2.2) > 1e-9):
            raise ValueError("Training data must use 1.0x ATR stops and 2.2x ATR targets")
        bar_index = int(meta.get("bar_index", -1))
        timestamp = meta.get("timestamp")
        timestamp_flags.add(bool(timestamp))
        if timestamp:
            parsed_timestamp = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
            if parsed_timestamp.tzinfo is None:
                raise ValueError("Training timestamps must include a timezone")
            parsed_timestamp = parsed_timestamp.astimezone(timezone.utc)
            if previous_timestamp is not None and parsed_timestamp < previous_timestamp:
                raise ValueError("JSONL rows must be sorted chronologically by timestamp")
            previous_timestamp = parsed_timestamp
        else:
            if bar_index <= previous_bar_index:
                raise ValueError("JSONL rows without timestamps need strictly increasing bar_index values")
            previous_bar_index = bar_index
        maximum_horizon = max(maximum_horizon, int(meta.get("horizon", 0)))
        maximum_horizon_minutes = max(
            maximum_horizon_minutes,
            int(meta.get("horizon_minutes", int(meta.get("horizon", 0)) * int(meta.get("bar_minutes", 1)))),
        )
        maximum_bar_minutes = max(maximum_bar_minutes, int(meta.get("bar_minutes", 1)))
        label = str(row["action"]).upper()
        if label not in LABEL2ID:
            raise ValueError(f"Unknown class label {label!r}; expected BUY, SELL, or HOLD")
        if label in {"BUY", "SELL"} and meta.get("reason") != f"{label.lower()}_target_before_stop":
            raise ValueError(f"{label} row is not marked as a target-before-stop outcome")

    if len(timestamp_flags) > 1:
        raise ValueError("Training JSONL cannot mix timestamped and untimestamped rows")

    eval_start = max(1, int(len(dataset) * (1.0 - args.eval_ratio)))
    if eval_start >= len(dataset):
        eval_start = len(dataset) - 1
    eval_rows = dataset.select(range(eval_start, len(dataset)))
    purge = max(args.purge_bars, maximum_horizon)
    train_end = eval_start
    if timestamp_flags == {True}:
        eval_first_timestamp = datetime.fromisoformat(
            str(eval_rows[0]["meta"]["timestamp"]).replace("Z", "+00:00")
        ).astimezone(timezone.utc)
        purge_minutes = max(
            maximum_horizon_minutes,
            args.purge_bars * max(1, maximum_bar_minutes),
        )
        cutoff = eval_first_timestamp - timedelta(minutes=purge_minutes)
        while train_end > 0:
            prior_timestamp = datetime.fromisoformat(
                str(dataset[train_end - 1]["meta"]["timestamp"]).replace("Z", "+00:00")
            ).astimezone(timezone.utc)
            if prior_timestamp < cutoff:
                break
            train_end -= 1
    else:
        eval_first_bar = int(eval_rows[0]["meta"]["bar_index"])
        while train_end > 0 and int(dataset[train_end - 1]["meta"]["bar_index"]) >= eval_first_bar - purge:
            train_end -= 1
    if train_end < 10 or len(eval_rows) < 2:
        raise ValueError(
            f"Chronological split leaves too little data after a {purge}-bar purge "
            f"(train={train_end}, validation={len(eval_rows)}); provide more history or reduce the horizon."
        )
    train_dataset = dataset.select(range(train_end))
    eval_dataset = eval_rows

    weight_totals = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
    weight_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
    for row in dataset:
        label = str(row["action"]).upper()
        weight_totals[label] += example_weight(
            label, row["meta"], args.r_weight_scale, args.hold_weight
        )
        weight_counts[label] += 1
    print("Loss weighting by label (mean weight x rows):")
    for label in ("BUY", "SELL", "HOLD"):
        if weight_counts[label]:
            mean = weight_totals[label] / weight_counts[label]
            print(f"  {label}: {mean:.3f} x {weight_counts[label]}")

    def encode_batch(batch):
        enc = tokenizer(batch["prompt"], truncation=True, max_length=args.max_length)
        enc["labels"] = [LABEL2ID[str(label).upper()] for label in batch["action"]]
        enc["weight"] = [
            example_weight(label, meta, args.r_weight_scale, args.hold_weight)
            for label, meta in zip(batch["action"], batch["meta"])
        ]
        return enc

    source_columns = dataset.column_names
    train_dataset = train_dataset.map(encode_batch, batched=True, remove_columns=source_columns)
    eval_dataset = eval_dataset.map(encode_batch, batched=True, remove_columns=source_columns)

    def compute_metrics(eval_pred):
        import numpy as np

        logits = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        predicted = np.asarray(logits).argmax(axis=-1)
        labels = np.asarray(eval_pred.label_ids)
        metrics = {}
        f1_values = []
        for class_id, class_name in ID2LABEL.items():
            true_positive = int(((predicted == class_id) & (labels == class_id)).sum())
            predicted_count = int((predicted == class_id).sum())
            support = int((labels == class_id).sum())
            precision = true_positive / predicted_count if predicted_count else 0.0
            recall = true_positive / support if support else 0.0
            metrics[f"{class_name.lower()}_precision"] = precision
            metrics[f"{class_name.lower()}_recall"] = recall
            metrics[f"{class_name.lower()}_support"] = support
            f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        metrics["macro_f1"] = float(np.mean(f1_values))
        return metrics

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
    # Newer transformers releases removed kwargs like warmup_ratio; drop
    # anything this version does not accept instead of failing at init.
    unsupported = [key for key in training_kwargs if key not in signature]
    for key in unsupported:
        training_kwargs.pop(key)
    if unsupported:
        print(f"Dropped TrainingArguments unsupported by this transformers version: {unsupported}")
    training_args = TrainingArguments(**training_kwargs)

    class WeightedCollator:
        """Pad token fields; carry per-example loss weights alongside."""

        def __init__(self, tokenizer) -> None:
            self.inner = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

        def __call__(self, features):
            weights = torch.tensor(
                [float(feature.pop("weight", 1.0)) for feature in features],
                dtype=torch.float32,
            )
            batch = self.inner(features)
            batch["weight"] = weights
            return batch

    class RWeightedTrainer(Trainer):
        """Cross-entropy weighted per example; weights are mean-normalized.

        The normalization keeps the weighted loss on the same scale as plain
        cross-entropy so the learning rate and cosine schedule stay valid.
        """

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            weights = inputs.pop("weight", None)
            labels = inputs["labels"]
            outputs = model(**inputs)
            losses = torch.nn.functional.cross_entropy(
                outputs.logits.view(-1, model.config.num_labels),
                labels.view(-1),
                reduction="none",
            )
            if weights is None:
                loss = losses.mean()
            else:
                loss = (losses * weights.to(losses.dtype)).sum() / weights.sum().clamp(min=1e-8)
            return (loss, outputs) if return_outputs else loss

    trainer = RWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=WeightedCollator(tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    validation_metrics = trainer.evaluate()
    print("Validation metrics (eval_loss is R-weighted cross-entropy):")
    for name, value in sorted(validation_metrics.items()):
        print(f"  {name}: {value}")
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    print(f"Saved Qwen sequence-classification LoRA adapter to {output.resolve()}")


if __name__ == "__main__":
    main()
