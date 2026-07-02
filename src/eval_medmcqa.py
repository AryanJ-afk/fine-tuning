"""
src/eval_medmcqa.py
Likelihood-based evaluation on MedMCQA.

For each question, we score the four options by the average log-probability the
model assigns to each option's text, and pick the highest. This works for BASE
(non-instruction-tuned) models because it only asks "which answer does the model
find most probable" -- no instruction-following needed.
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from peft import PeftModel

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_medmcqa(n=None, split="validation"):
    """Load MedMCQA. 'validation' split (test labels are hidden). n = take a subset."""
    dataset = load_dataset("openlifescienceai/medmcqa", split=split)
    if n is not None:
        dataset = dataset.select(range(n))
    return dataset


def score_option(model, tokenizer, prompt, option, device):
    """Average log-prob the model gives to `option` following `prompt`. Higher = more likely."""
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(prompt + " " + option, return_tensors="pt").input_ids.to(device)

    # Tokens belonging to the option = everything after the prompt.
    option_len = full_ids.shape[1] - prompt_ids.shape[1]
    if option_len <= 0:
        return float("-inf")

    with torch.no_grad():
        logits = model(full_ids).logits                 # [1, seq_len, vocab]

    log_probs = torch.log_softmax(logits, dim=-1)

    # logits at position i predict the token at position i+1, so shift by one:
    # for every real token, pull out the log-prob the model assigned it.
    token_log_probs = log_probs[0, :-1].gather(
        1, full_ids[0, 1:].unsqueeze(1)
    ).squeeze(1)                                          # [seq_len - 1]

    # The option's tokens are the last `option_len` of these. Average them
    # (length-normalised, so longer options aren't unfairly penalised).
    return token_log_probs[-option_len:].mean().item()


def predict(model, tokenizer, example, device):
    """Index (0-3) of the highest-scoring option."""
    prompt = f"Question: {example['question']}\nAnswer:"
    options = [example["opa"], example["opb"], example["opc"], example["opd"]]
    scores = [score_option(model, tokenizer, prompt, opt, device) for opt in options]
    return int(torch.tensor(scores).argmax())


def evaluate(model, tokenizer, dataset, device):
    """MCQ accuracy. cop is 0-indexed: 0=opa, 1=opb, 2=opc, 3=opd."""
    model.eval()
    correct = 0
    for example in tqdm(dataset, desc="Evaluating"):
        if predict(model, tokenizer, example, device) == example["cop"]:
            correct += 1
    return correct / len(dataset)

def load_adapter_model(base_name, adapter_path, device):
    """Load a base model in 4-bit, then attach a trained LoRA/QLoRA adapter on top."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_path)   # <-- attaches the adapter
    return model


if __name__ == "__main__":
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base")
    model = load_adapter_model("Qwen/Qwen3-4B-Base", "/content/drive/MyDrive/stage2_model", device)

    acc = evaluate(model, tokenizer, load_medmcqa(n=500), device)
    print(f"QLoRA MedMCQA accuracy: {acc:.3f}")