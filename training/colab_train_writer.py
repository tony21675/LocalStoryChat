#!/usr/bin/env python3
"""
LocalStoryChat Qwen3.5-9B writer training for Google Colab.

Run this from a Colab T4 GPU after installing Unsloth.

The script:
1. Resets the temporary Colab project folder.
2. Clones the LocalStoryChat training branch.
3. Rebuilds and validates the 48-example Gold dataset.
4. Loads Qwen3.5-9B-Base in 4-bit with Unsloth.
5. Adds language-side LoRA adapters only.
6. Fine-tunes conversational examples with TRL SFTTrainer.
7. Trains on assistant responses only.
8. Saves the LoRA adapter and tokenizer and creates a ZIP.
"""

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

    if "T4" not in name:
        print("WARNING: this script was tuned for a 16 GB-class T4 runtime.")


def prepare_project():
    if PROJECT.exists():
        print(f"Removing previous temporary project: {PROJECT}")
        shutil.rmtree(PROJECT)

    run([
        "git",
        "clone",
        "--branch", BRANCH,
        "--single-branch",
        REPO,
        str(PROJECT),
    ])

    os.chdir(PROJECT)
    run(["python", "training/prepare_gold_dataset.py"])


def load_dataset():
    from datasets import load_dataset

    path = str(PROJECT / "training" / "gold_dataset.jsonl")
    dataset = load_dataset("json", data_files=path, split="train")

    if len(dataset) != 48:
        raise ValueError(f"Expected 48 examples, found {len(dataset)}")

    # Tell the Qwen3.5 chat template to render ordinary prose, not thinking traces.
    dataset = dataset.map(
        lambda row: {"chat_template_kwargs": {"enable_thinking": False}},
        num_proc=1,
    )

    print(f"Loaded {len(dataset)} Gold examples")
    return dataset


def main():
    print("=" * 70)
    print("LocalStoryChat Qwen3.5-9B writer training")
    print("=" * 70)

    check_gpu()
    prepare_project()

    import torch
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    dataset = load_dataset()

    print("\nLoading Qwen3.5-9B-Base in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_LENGTH,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("\nAttaching language-side LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        finetune_vision_layers=False,
    )

    print("\nLoRA attached successfully.")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"GPU memory allocated: {allocated:.2f} GiB")
        print(f"GPU memory reserved: {reserved:.2f} GiB")

    split = dataset.train_test_split(test_size=5, seed=SEED)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        max_length=MAX_LENGTH,
        dataset_num_proc=1,
        assistant_only_loss=True,
        packing=False,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=2,
        fp16=True,
        bf16=False,
        tf32=False,
        optim="adamw_8bit",
        weight_decay=0.0,
        seed=SEED,
        report_to="none",
        remove_unused_columns=False,
    )

    print("\nStarting SFT training...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    print("\nSaving LoRA adapter...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    archive = shutil.make_archive(
        "/content/qwen3_5_9b_writer_colab",
        "zip",
        str(OUTPUT_DIR),
    )

    print("\nTRAINING COMPLETE")
    print(f"Adapter directory: {OUTPUT_DIR}")
    print(f"Adapter ZIP: {archive}")


if __name__ == "__main__":
    main()
