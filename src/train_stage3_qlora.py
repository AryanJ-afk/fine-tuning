"""
src/train_stage3_qlora.py
Stage 3: QLoRA instruction-tuning of Qwen3-4B-Base on AlpaCare.
Base loaded in 4-bit (frozen); only small LoRA adapters are trained.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from data_prep_sft import load_alpacare


def main():
    model_name = "Qwen/Qwen3-4B-Base"

    data = load_alpacare().shuffle(seed=42).select(range(1000))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit quant. compute dtype = bf16 (Qwen3 is bf16-native; matching it avoids dtype clashes).
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir="/content/stage3_out",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        warmup_steps=10,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,                 # bf16 training -> NO grad scaler -> the bf16 error can't occur
        fp16=False,
        dataloader_pin_memory=False,
        report_to="none",
        max_length=256,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=data,
        peft_config=lora_config,
    )

    trainer.train()
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    trainer.save_model("/content/drive/MyDrive/stage3_model")
    tokenizer.save_pretrained("/content/drive/MyDrive/stage3_model")
    print("Saved adapter to /content/drive/MyDrive/stage3_model")


if __name__ == "__main__":
    main()