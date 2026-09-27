#!/usr/bin/env python3
"""
LocalStoryChat Qwen3.5-9B writer training for Google Colab.

Designed to be run ONCE from a Colab GPU runtime after Unsloth is installed.
It deliberately avoids Axolotl and avoids notebook-local git state.

The script:
1. Resets the temporary Colab project folder.
2. Clones the LocalStoryChat branch.
3. Rebuilds and validates the 48-example Gold dataset.
4. Converts each user/assistant example into a prompt/completion record.
5. Loads Qwen3.5-9B-Base with Unsloth 4-bit QLoRA.
6. Trains only language layers. Vision layers remain frozen.
7. Saves the LoRA adapter and tokenizer.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

PROJECT = Path("/content/LocalStoryChat")
BRANCH = "generic-story-engine-rebuild"
REPO = "https://github.com/tony21675/LocalStoryChat.git"
MODEL_NAME = "Qwen/Qwen3.5-9B-Base"
OUTPUT_DIR = PROJECT / "training" / "outputs" / "qwen3_5_9b_writer_colab"
MAX_LENGTH = 2048
SEED = 42


def run(cmd):
    print("\n$", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def check_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU is available. In Colab choose Runtime -> Change runtime type -> T4 GPU."
        )
    name = torch.cuda.get_device_name(0)
    memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {name}")
    print(f"VRAM: {memory:.2f} GiB")
    if memory < 14:
        print("WARNING: this GPU has less than 14 GiB VRAM; the conservative T4 settings may not fit.")


def prepare_project():
    if PROJECT.exists():
        shutil.rmtree(PROJECT)
    run([
        "git", "clone",
        "--branch", BRANCH,
        "--single-branch",
        REPO,
        str(PROJECT),
    ])
    os.chdir(PROJECT)
    run(["python", "training/prepare_gold_dataset.py"])


def load_examples():
    path = PROJECT / "training" / "gold_dataset.jsonl"
    rows = []

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue

        data = json.loads(line)
        messages = data.get("messages", [])

        if len(messages) != 2:
            raise ValueError(f"Dataset line {line_no}: expected exactly 2 messages")

        if messages[0].get("role") != "user":
            raise ValueError(f"Dataset line {line_no}: first message must be user")

        if messages[1].get("role") != "assistant":
            raise ValueError(f"Dataset line {line_no}: second message must be assistant")

        rows.append({
            "prompt": [
                {"role": "user", "content": messages[0]["content"]}
            ],
            "completion": [
                {"role": "assistant", "content": messages[1]["content"]}
            ],
        })

    if len(rows) != 48:
        raise ValueError(f"Expected 48 examples, found {len(rows)}")

    print(f"Prepared {len(rows)} training examples")
    return rows


def tokenize_examples(rows, tokenizer):
    from datasets import Dataset

    processed = []

    for row in rows:
        prompt = row["prompt"]
        full = prompt + row["completion"]

        prompt_ids = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
        )

        full_ids = tokenizer.apply_chat_template(
            full,
            tokenize=True,
            add_generation_prompt=False,
        )

        # Chat-template implementations should preserve the prompt prefix.
        # Use the actual longest common prefix rather than assuming exact equality.
        common = 0
        limit = min(len(prompt_ids), len(full_ids))
        while common < limit and prompt_ids[common] == full_ids[common]:
            common += 1

        if common == 0:
            raise ValueError("Could not align prompt and completion tokens")

        if common < len(prompt_ids):
            print(
                f"Warning: chat-template prefix differed by {len(prompt_ids) - common} tokens; using common prefix",
                flush=True,
            )

        labels = ([-100] * common) + full_ids[common:]

        if len(full_ids) > MAX_LENGTH:
            full_ids = full_ids[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]

        processed.append({
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": [1] * len(full_ids),
        })

    return processed


def main():
    print("=" * 70)
    print("LocalStoryChat writer training")
    print("=" * 70)

    check_gpu()
    prepare_project()

    import torch
    from datasets import Dataset

    rows = load_examples()

    print("\nLoading Unsloth and Qwen3.5-9B-Base...")
    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_LENGTH,
        load_in_4bit=True,
        load_in_8bit=False,
        load_in_16bit=False,
        full_finetuning=False,
        text_only=True,
    )

    print("\nAttaching language-only LoRA adapter...")
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        random_state=SEED,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenized = tokenize_examples(rows, tokenizer)
    dataset = Dataset.from_list(tokenized)

    split = dataset.train_test_split(test_size=5, seed=SEED)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    class Collator:
        def __call__(self, features):
            max_len = max(len(x["input_ids"]) for x in features)
            pad_id = tokenizer.pad_token_id

            input_ids = []
            labels = []
            attention_mask = []

            for x in features:
                pad = max_len - len(x["input_ids"])
                input_ids.append(x["input_ids"] + [pad_id] * pad)
                labels.append(x["labels"] + [-100] * pad)
                attention_mask.append(x["attention_mask"] + [0] * pad)

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }

    from transformers import Trainer, TrainingArguments

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.0,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=2,
        fp16=True,
        bf16=False,
        tf32=False,
        gradient_checkpointing=True,
        optim="adamw_8bit",
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        remove_unused_columns=False,
    )

    print("\nStarting training...")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Collator(),
    )

    trainer.train(resume_from_checkpoint=False)

    print("\nSaving LoRA adapter...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    trainer.save_state()

    print("\nTRAINING COMPLETE")
    print(f"Adapter directory: {OUTPUT_DIR}")

    archive_base = Path("/content/qwen3_5_9b_writer_colab")
    archive = shutil.make_archive(
        str(archive_base),
        "zip",
        str(OUTPUT_DIR),
    )
    print(f"Adapter ZIP: {archive}")


if __name__ == "__main__":
    main()
