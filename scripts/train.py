"""QLoRA SFT for the Jev BUY/SELL/HOLD readout on Qwen2.5-3B-Instruct.

Fits an affordable single-GPU pod (L4 / A10 / RTX 4090 24GB; T4 16GB with
``--per-device-train-batch-size 1``). Loads ``data/jev_train_5m.jsonl`` as
prompt-completion pairs so TRL masks the user state and trains only the
discrete action token.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_DATA = "data/jev_train_5m.jsonl"
DEFAULT_OUTPUT = "models/checkpoints/qwen25-3b-jev-lora"

# Qwen2 / Qwen2.5 attention projections.
ATTN_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--data", default=DEFAULT_DATA, help="Jev JSONL from dataset_generator.py")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report-to",
        default="none",
        help="Trainer logger: none | wandb | tensorboard",
    )
    return parser.parse_args()


def compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_jev_dataset(path: str, eval_ratio: float, seed: int):
    jsonl = Path(path)
    if not jsonl.is_absolute():
        jsonl = PROJECT_ROOT / jsonl
    if not jsonl.is_file():
        raise FileNotFoundError(
            f"Training file not found: {jsonl}. Generate it with dataset_generator.py"
        )

    raw = load_dataset("json", data_files=str(jsonl), split="train")

    def to_prompt_completion(example: dict) -> dict:
        messages = example.get("messages") or []
        user = None
        if messages:
            user = next(
                (m["content"] for m in messages if m.get("role") == "user"),
                None,
            )
        if not user:
            user = example.get("prompt")
        action = example.get("action") or str(example.get("completion", "")).strip()
        if not user or not action:
            raise ValueError("JSONL row missing prompt/user content or action label")
        return {
            "prompt": [{"role": "user", "content": user}],
            "completion": [{"role": "assistant", "content": action}],
        }

    formatted = raw.map(to_prompt_completion, remove_columns=raw.column_names)
    if eval_ratio and 0.0 < eval_ratio < 1.0 and len(formatted) > 20:
        split = formatted.train_test_split(test_size=eval_ratio, seed=seed)
        return split["train"], split["test"]
    return formatted, None


def build_bnb_config(dtype: torch.dtype) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )


def build_lora_config(r: int, alpha: int, dropout: float) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(ATTN_TARGET_MODULES),
    )


def build_sft_config(args: argparse.Namespace, dtype: torch.dtype, has_eval: bool) -> SFTConfig:
    use_bf16 = dtype == torch.bfloat16
    kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=args.save_steps if has_eval else None,
        bf16=use_bf16,
        fp16=not use_bf16,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        max_length=args.max_length,
        packing=False,
        completion_only_loss=True,
        report_to=args.report_to,
        seed=args.seed,
        model_init_kwargs={
            "device_map": "auto",
            "trust_remote_code": True,
        },
    )
    if has_eval:
        kwargs["load_best_model_at_end"] = True
        kwargs["metric_for_best_model"] = "eval_loss"
        kwargs["greater_is_better"] = False
    else:
        kwargs.pop("eval_steps")
    try:
        return SFTConfig(**kwargs)
    except TypeError:
        kwargs.pop("max_length", None)
        kwargs["max_seq_length"] = args.max_length
        if "eval_strategy" in kwargs:
            kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
        kwargs.pop("gradient_checkpointing_kwargs", None)
        kwargs.pop("completion_only_loss", None)
        kwargs.pop("model_init_kwargs", None)
        return SFTConfig(**kwargs)


def main() -> None:
    global torch, load_dataset, LoraConfig, AutoTokenizer, BitsAndBytesConfig, SFTConfig, SFTTrainer

    args = parse_args()
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    args.data = str(data_path)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)
    dtype = compute_dtype()
    train_ds, eval_ds = load_jev_dataset(args.data, args.eval_ratio, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    sft_args = build_sft_config(args, dtype, has_eval=eval_ds is not None)
    lora_config = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)
    bnb_config = build_bnb_config(dtype)

    trainer_kwargs = dict(
        model=args.model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    try:
        trainer = SFTTrainer(quantization_config=bnb_config, **trainer_kwargs)
    except TypeError:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model.config.use_cache = False
        fallback = {k: v for k, v in trainer_kwargs.items() if k != "model"}
        if "processing_class" in fallback:
            fallback["tokenizer"] = fallback.pop("processing_class")
        trainer = SFTTrainer(model=model, **fallback)

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
