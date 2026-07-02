# Biomedical LLM Fine-Tuning: Full FT, LoRA & QLoRA

A hands-on demonstration of three fine-tuning techniques on the biomedical domain, using the HuggingFace stack (`transformers`, `peft`, `trl`, `bitsandbytes`, `datasets`), evaluated with a leakage-free likelihood-based harness on MedMCQA.

The goal is to demonstrate the techniques end-to-end and compare them fairly.

## What this is (and isn't)

This is **three technique demos**, not one model flowing through a pipeline:

- **Stage 1 — Full fine-tuning (continued pretraining)** of `Qwen3-0.6B-Base` on PubMed abstracts. A small model is used because full FT of a larger one doesn't fit on a free GPU — which is itself the motivation for PEFT.
- **Stage 2 — LoRA** instruction-tuning of `Qwen3-4B-Base` on AlpaCare-MedInstruct (16-bit frozen base).
- **Stage 3 — QLoRA** instruction-tuning of the *same* `Qwen3-4B-Base` on the *same* data (4-bit frozen base).

Stages 2 and 3 use identical data, LoRA config, and hyperparameters — the only difference is 4-bit quantization — so their comparison is controlled and valid.

## Data

| Use | Dataset |
|---|---|
| Continued pretraining (Stage 1) | `casinca/PUBMED_title_abstracts_2019_baseline` |
| Instruction tuning (Stages 2 & 3) | `lavita/AlpaCare-MedInstruct-52k` |
| Evaluation | `openlifescienceai/medmcqa` (validation split) |

Training on AlpaCare and evaluating on MedMCQA (different sources) keeps the eval leakage-free.

## Evaluation

Likelihood-based multiple-choice accuracy: for each question, the four options are scored by the model's average log-probability and the highest is chosen. This works for base (non-instruction-tuned) models and needs no output parsing. Random baseline = 25%.

## Results

**Stage 1 — continued pretraining (Qwen3-0.6B-Base), 500 questions:**

| | MedMCQA |
|---|---|
| Base | 0.278 |
| + continued pretraining | 0.296 |

**Stages 2 & 3 — LoRA vs QLoRA (Qwen3-4B-Base), 500 questions:**

| Model | MedMCQA | Peak VRAM |
|---|---|---|
| Base | 0.358 | — |
| LoRA (16-bit base) | 0.342 | 11.13 GB |
| QLoRA (4-bit base) | 0.350 | 5.39 GB |

## Findings

- **Accuracy differences are within noise.** With small training subsets on already-capable base models, none of the tuning runs shifted MedMCQA accuracy meaningfully. This is expected and reported honestly.
- **QLoRA matched LoRA's accuracy at substantially lower memory** (4-bit base ≈ 5.39 GB peak) — demonstrating QLoRA's core value: the memory savings come at no accuracy cost.

## Stack

`transformers`, `peft`, `trl` (`SFTTrainer`), `bitsandbytes` (4-bit NF4), `datasets`, PyTorch. Trained on a Colab T4 (16 GB).

## Structure

```
src/
  data_prep.py         # Stage 1: load, tokenize, chunk PubMed
  data_prep_sft.py     # Stages 2/3: format AlpaCare (Alpaca-style)
  eval_medmcqa.py      # likelihood-based MedMCQA harness
  train_stage1.py      # full fine-tuning
  train_stage3_qlora.py# QLoRA (Stage 2 is this with quantization removed)
```

## Notes

- On a T4, use `bf16` (not `fp16`) with bf16-native models like Qwen3 — bf16 needs no gradient scaler and avoids the fp16-scaler/bf16 incompatibility.
- The standard OOM-fix ladder: smaller batch size → gradient accumulation → gradient checkpointing → quantization.
