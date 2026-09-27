# LocalStoryChat Writer Training

## Target

The first trained writer is based on Qwen/Qwen3.5-9B-Base and uses QLoRA.

The goal is not to teach the model this novel's canon. The goal is to teach reusable writing behavior:

- contemporary American English
- natural dialogue
- grounded description
- believable pacing
- consistent point of view
- subtext without over-explaining
- realistic character interaction
- scene boundaries
- restrained emotion
- grounded adult romance when appropriate

Novel facts remain in the story JSON/state system.

## Training data

gold_dataset.jsonl contains the 48 reusable Gold examples.

The Gold examples use the OpenAI Messages format:

- user prompt
- assistant target prose

Only assistant messages are trained.

## First-pass config

Use qwen3_5_9b_qlora.yaml.

This is intentionally conservative because 48 examples is a small style dataset. The first run is an experiment, not the final writer.

## Hardware

Do not try to fine-tune Qwen3.5-9B on the 8 GB Surface Laptop 3.

Run QLoRA on a CUDA GPU in the cloud or on a more capable local machine. A 24 GB+ GPU is a practical target for leaving memory headroom, although actual usage depends on software versions and training settings.

## Important Qwen3.5 details

Qwen3.5 is a multimodal architecture. This text-only training config keeps the multimodal modules frozen and uses the Qwen3.5 chat template.

The config disables thinking in the chat template because the target is ordinary novel prose, not reasoning traces.

Sample packing is disabled for the first run. This also avoids a known Qwen3.5/Axolotl failure mode reported when packing was enabled in some setups.

## First run

From the repository root:

axolotl train training/qwen3_5_9b_qlora.yaml

The training environment should be created on the GPU machine, not on the Surface.

## After training

The important artifact is the LoRA adapter under:

training/outputs/qwen3_5_9b_writer/

Do not immediately discard the adapter or merge it. First test the adapter against a fixed set of writing prompts.

Compare:

1. Qwen3.5-9B-Base
2. base + writer adapter
3. the same prompts with LocalStoryChat story context

The comparison should look for actual improvements in natural prose, dialogue, pacing, and continuity behavior without teaching the model novel-specific facts.

## What success looks like

A successful first pass should make the model sound more like a consistent contemporary fiction writer without forcing every scene into the same structure.

It should still be able to write quiet scenes, dialogue-heavy scenes, action, emotional moments, and grounded adult romance.

If it becomes repetitive, overly short, overly sentimental, or starts copying the Gold examples, we adjust the dataset/training settings rather than piling on more system-prompt rules.


## Colab: current start-here path

Use only:

training/START_HERE_colab_writer.ipynb

This replaces the older Axolotl/Kaggle notebook experiments.

The clean Colab flow is:

1. Select a free T4 GPU in Runtime -> Change runtime type.
2. Run the GPU check.
3. Run the single Unsloth installation cell.
4. Run the single training cell.
5. Download /content/qwen3_5_9b_writer_colab.zip when training finishes.

The training cell downloads the current training script directly from this branch and the script resets its temporary Colab workspace before cloning, so stale Colab folders cannot break the run.

The training implementation uses Unsloth + Hugging Face Trainer rather than Axolotl. The 48 Gold examples are converted to prompt/completion pairs, with loss applied to the completion only. Vision layers stay frozen and only language-side LoRA adapters are trained.
