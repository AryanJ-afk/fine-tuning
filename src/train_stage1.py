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

    raw = load_pubmed_corpus(n=50000)                      # raw abstracts
    tokenized = raw.map(
        lambda x: tokenize_corpus(x, tokenizer),
        batched=True,
        remove_columns=raw.column_names,
    )
    blocks = tokenized.map(group_into_blocks, batched=True)  # 1024-token blocks w/ labels
    print(blocks)                                            # sanity-check the block count

    # --- Model ---
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # --- Collator: batches the blocks; mlm=False = causal LM (next-token) ---
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- Training config ---
    args = TrainingArguments(
        output_dir="./stage1_out",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,        # effective batch = 32
        learning_rate=2e-5,
        warmup_ratio=0.03,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,                           # MPS doesn't use fp16/bf16 flags; leave both off
        bf16=False,
        report_to="none",                     # no W&B yet
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