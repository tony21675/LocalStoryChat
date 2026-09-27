#!/usr/bin/env python3
import os, shutil, subprocess
from pathlib import Path

PROJECT=Path("/content/LocalStoryChat")
BRANCH="generic-story-engine-rebuild"
REPO="https://github.com/tony21675/LocalStoryChat.git"
MODEL="Qwen/Qwen3.5-9B-Base"

if not __import__("torch").cuda.is_available():
    raise RuntimeError("CUDA GPU not available.")

if PROJECT.exists():
    shutil.rmtree(PROJECT)

subprocess.run(
    ["git","clone","--branch",BRANCH,"--single-branch",REPO,str(PROJECT)],
    check=True,
)

os.chdir(PROJECT)
subprocess.run(["python","training/prepare_gold_dataset.py"],check=True)

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel

print("Loading model in 4-bit...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=2048,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id=tokenizer.eos_token_id

print("Attaching language-only LoRA...")
model=FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    finetune_vision_layers=False,
)

data=load_dataset(
    "json",
    data_files=str(PROJECT/"training"/"gold_dataset.jsonl"),
    split="train",
)

sample=data[0]["messages"]
rendered=tokenizer.apply_chat_template(
    sample,
    tokenize=False,
    add_generation_prompt=False,
    enable_thinking=False,
)

tokens=tokenizer(rendered,return_tensors="pt")["input_ids"]
print("First example tokens:", tokens.shape[-1])

allocated=torch.cuda.memory_allocated(0)/(1024**3)
reserved=torch.cuda.memory_reserved(0)/(1024**3)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory/(1024**3):.2f} GiB")
print(f"VRAM allocated after model+LoRA: {allocated:.2f} GiB")
print(f"VRAM reserved after model+LoRA: {reserved:.2f} GiB")
print("PREFLIGHT PASSED")
print("No training was performed.")
