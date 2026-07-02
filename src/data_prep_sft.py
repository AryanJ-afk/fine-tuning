"""
src/data_prep_sft.py
Stage 2/3 data prep: format AlpaCare-MedInstruct-52k into Alpaca-style instruction text.
The SAME prepared data feeds BOTH the LoRA (Stage 2) and QLoRA (Stage 3) runs --
that's what makes their comparison valid.
"""

from datasets import load_dataset


# Standard Alpaca prompt templates: one for rows that have an 'input', one for rows that don't.
PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
)

PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n{output}"
)


def format_example(example):
    """Turn one {instruction, input, output} row into a single Alpaca-style 'text' string."""
    inp = (example["input"] or "").strip()
    if inp and inp.lower() != "<noinput>":      # <noinput> is AlpaCare's "no input" sentinel
        text = PROMPT_WITH_INPUT.format(
            instruction=example["instruction"],
            input=example["input"],
            output=example["output"],
        )
    else:
        text = PROMPT_NO_INPUT.format(
            instruction=example["instruction"],
            output=example["output"],
        )
    return {"text": text}


def load_alpacare(n=None):
    """Load AlpaCare, format into a single 'text' column ready for SFTTrainer. n = take a subset."""
    dataset = load_dataset("lavita/AlpaCare-MedInstruct-52k", split="train")
    if n is not None:
        dataset = dataset.select(range(n))
    return dataset.map(format_example, remove_columns=dataset.column_names)
