"""
src/train_stage1.py
Stage 1: continued pretraining (full fine-tuning) of Qwen3-0.6B-Base on PubMed abstracts.
Takes the base model and adapts it to biomedical text via plain next-token prediction.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from data_prep import load_pubmed_corpus, tokenize_corpus, group_into_blocks


def main():
    model_name = "Qwen/Qwen3-0.6B-Base"

    # --- Data: the three functions you built, chained together ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    raw = load_pubmed_corpus(n=10000)                      # raw abstracts
    tokenized = raw.map(
        lambda x: tokenize_corpus(x, tokenizer),
        batched=True,
        remove_columns=raw.column_names,
    )
    blocks = tokenized.map(group_into_blocks, batched=True)  # 1024-token blocks w/ labels
    print(blocks)                                            # sanity-check the block count

    # --- Model ---
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

    # --- Collator: batches the blocks; mlm=False = causal LM (next-token) ---
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- Training config ---
    args = TrainingArguments(
    output_dir="./stage1_out",
    num_train_epochs=1,
    per_device_train_batch_size=1,          # was 4 — the big lever
    gradient_accumulation_steps=32,         # keep effective batch = 32
    gradient_checkpointing=True,            # recompute activations instead of storing them
    gradient_checkpointing_kwargs={"use_reentrant": False},  # avoids a warning in newer transformers
    learning_rate=2e-5,
    warmup_steps=5,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    fp16=True,                              # ← confirm this is actually True (halves activation memory)
    dataloader_pin_memory=False,
    report_to="none",
)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=blocks,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model("./stage1_model")      # saves the adapted model + config
    tokenizer.save_pretrained("./stage1_model")
    print("Saved to ./stage1_model")


if __name__ == "__main__":
    main()