#!/usr/bin/env python3
"""
LocalStoryChat Qwen3.5-9B writer training for Google Colab.

Uses the current Unsloth + TRL SFT pattern for Qwen3.5:
- format the chat conversations with Qwen3.5's chat template
- train a text field with SFTTrainer
- language-side LoRA only
- 4-bit loading
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
MAX_SEQ_LENGTH = 2048
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
    memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"GPU: {name}")
    print(f"VRAM: {memory:.2f} GiB")


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

    print(f"Loaded {len(dataset)} Gold examples")
    return dataset


def format_dataset(dataset, tokenizer):
    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
            )
            for convo in convos
        ]
        return {"text": texts}

    return dataset.map(
        formatting_prompts_func,
        batched=True,
        num_proc=1,
    )


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
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )

    tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Formatting Gold conversations...")
    dataset = format_dataset(dataset, tokenizer)

    print("Attaching language-only LoRA...")
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

    split = dataset.train_test_split(test_size=5, seed=SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=2,
        packing=False,
        fp16=False,
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
        tokenizer=tokenizer,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        args=training_args,
    )

    trainer.train()

    print("\nSaving LoRA adapter...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    archive = shutil.make_archive(
        "/content/qwen3_5_9b_writer",
        "zip",
        str(OUTPUT_DIR),
    )

    print("\nTRAINING COMPLETE")
    print(f"Adapter directory: {OUTPUT_DIR}")
    print(f"Adapter ZIP: {archive}")


if __name__ == "__main__":
    main()
