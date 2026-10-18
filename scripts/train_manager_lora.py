"""QLoRA fine-tuning for the portfolio-manager LLM (PolicyWeights proposer).

Reads the JSONL produced by scripts/generate_manager_dataset.py
({prompt, completion, meta}) and trains a causal-LM LoRA with loss on the
completion only. The chronological split uses meta.window_end_bar and purges
at least the reward horizon, mirroring train_lora.py's leakage discipline.

Unlike train_lora.py this script is self-contained: it must stay importable
on a fresh clone that predates core/manager.py, so it only needs
transformers/peft/datasets and the JSONL file.

Run on a CUDA GPU (Kaggle T4 is sufficient for the 1.5B default):
    python scripts/train_manager_lora.py \
        --data data/manager_train.jsonl --output-dir models/manager_adapter
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
WEIGHT_KEYS = ("trend", "stretch", "shock", "vol_regime")
JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/manager_train.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Qwen2.5-0.5B/1.5B/3B-Instruct all work; 1.5B is plenty for this task")
    parser.add_argument("--output-dir", default="models/manager_adapter")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--eval-ratio", type=float, default=0.15)
    parser.add_argument("--purge-bars", type=int, default=24,
                        help="Chronological gap before validation; at least the reward horizon")
    parser.add_argument("--max-eval-samples", type=int, default=32,
                        help="How many validation prompts to run the generation check on")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rows(data_path: Path) -> list[dict]:
    rows = []
    with data_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for field in ("prompt", "completion", "meta"):
                if field not in row:
                    raise ValueError(f"{data_path}:{line_number} missing field {field!r}")
            meta = row["meta"]
            if "source" not in meta or "window_end_bar" not in meta:
                raise ValueError(f"{data_path}:{line_number} meta needs source and window_end_bar")
            for key in WEIGHT_KEYS:
                if key not in json.loads(row["completion"]):
                    raise ValueError(f"{data_path}:{line_number} completion missing {key!r}")
            rows.append(row)
    if len(rows) < 20:
        raise ValueError(f"Need at least 20 rows, found {len(rows)}")
    return rows


def chronological_split(rows: list[dict], eval_ratio: float, purge_bars: int) -> tuple[list[dict], list[dict]]:
    """Sort by (real-before-synthetic, window_end_bar), split with a purge gap."""
    order = {"real_mnq": 0, "synthetic": 1}

    def sort_key(row: dict) -> tuple[int, int]:
        meta = row["meta"]
        return (order.get(str(meta["source"]), 1), int(meta["window_end_bar"]))

    rows = sorted(rows, key=sort_key)
    previous_key: tuple[int, int] | None = None
    horizon = 1
    for row in rows:
        key = sort_key(row)
        if previous_key is not None and key[0] == previous_key[0] and key[1] < previous_key[1]:
            raise ValueError("window_end_bar must be non-decreasing within a source")
        previous_key = key
        horizon = max(horizon, int(row["meta"].get("forward_horizon", 1)))
    purge = max(purge_bars, horizon + 1)

    eval_start = max(1, int(len(rows) * (1.0 - eval_ratio)))
    eval_rows = rows[eval_start:]
    if len(eval_rows) < 2:
        raise ValueError("eval ratio too small; need at least two validation rows")
    first_eval_key = sort_key(eval_rows[0])
    train_end = eval_start
    while train_end > 0 and sort_key(rows[train_end - 1])[0] == first_eval_key[0] and (
        sort_key(rows[train_end - 1])[1] >= first_eval_key[1] - purge
    ):
        train_end -= 1
    train_rows = rows[:train_end]
    if len(train_rows) < 10:
        raise ValueError(
            f"Chronological split leaves too little training data after a {purge}-bar purge "
            f"(train={len(train_rows)}, validation={len(eval_rows)})"
        )
    return train_rows, eval_rows


def extract_weights(text: str) -> dict[str, float] | None:
    """First JSON object in text -> {key: float}, or None (mirrors core parse)."""
    match = JSON_OBJECT_RE.search(text)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    weights: dict[str, float] = {}
    for key in WEIGHT_KEYS:
        try:
            weights[key] = max(0.0, float(payload.get(key, 0.0)))
        except (TypeError, ValueError):
            return None
    total = sum(weights.values())
    if total <= 0:
        return None
    return {key: weights[key] / total for key in WEIGHT_KEYS}


def main() -> None:
    global torch
    args = parse_args()
    import numpy as np
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    if args.max_length < 64 or args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("max_length, batch_size, and gradient accumulation must be positive")
    if not 0 < args.eval_ratio < 1:
        raise ValueError("eval_ratio must be > 0 and < 1")
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    args.data = str(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Manager training JSONL not found: {data_path}")
    output_path = Path(args.output_dir)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    args.output_dir = str(output_path)

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires a CUDA GPU; run this on a GPU pod/Kaggle GPU runtime")

    rows = load_rows(data_path)
    train_rows, eval_rows = chronological_split(rows, args.eval_ratio, args.purge_bars)
    print(f"rows: {len(rows)} total | train={len(train_rows)} eval={len(eval_rows)} (purged split)")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def encode_row(row: dict) -> dict:
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(row["completion"], add_special_tokens=False)["input_ids"]
        target_ids = target_ids + [tokenizer.eos_token_id]
        # Tokenizer boundary safety: find how much of the prompt survives as a
        # prefix of the concatenated sequence, and mask exactly that much.
        prefix = 0
        for a, b in zip(prompt_ids, prompt_ids + target_ids):
            if a == b:
                prefix += 1
            else:
                break
        input_ids = (prompt_ids + target_ids)[: args.max_length]
        labels = ([-100] * min(prefix, len(input_ids))) + input_ids[min(prefix, len(input_ids)):]
        return {"input_ids": input_ids, "labels": labels}

    train_dataset = Dataset.from_list([encode_row(row) for row in train_rows])
    eval_dataset = Dataset.from_list([encode_row(row) for row in eval_rows])

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
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
    import inspect

    eval_key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters else "evaluation_strategy"
    training_kwargs[eval_key] = "epoch"
    training_args = TrainingArguments(**training_kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
        processing_class=tokenizer,
    )
    trainer.train()
    eval_metrics = trainer.evaluate()
    print("Validation metrics (eval_loss is completion token cross-entropy):")
    for name, value in sorted(eval_metrics.items()):
        print(f"  {name}: {value}")

    # Generation check: can the adapter actually emit parseable weights, and
    # how far are they from the teacher targets? This matters more than loss.
    model.eval()
    model.config.use_cache = True
    sample = eval_rows[: max(2, min(args.max_eval_samples, len(eval_rows)))]
    valid = 0
    errors: list[float] = []
    for row in sample:
        inputs = tokenizer(row["prompt"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs, max_new_tokens=64, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        reply = tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        predicted = extract_weights(reply)
        if predicted is None:
            continue
        valid += 1
        target = extract_weights(row["completion"])
        if target is not None:
            errors.append(
                float(np.mean([abs(predicted[key] - target[key]) for key in WEIGHT_KEYS]))
            )
    print(
        f"Generation check on {len(sample)} prompts: "
        f"valid JSON {valid}/{len(sample)} ({valid / max(len(sample), 1):.0%}), "
        f"mean weight MAE {float(np.mean(errors)) if errors else float('nan'):.4f}"
    )

    output_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(output_path)
    print(f"Saved manager LoRA adapter to {output_path.resolve()}")


if __name__ == "__main__":
    main()
